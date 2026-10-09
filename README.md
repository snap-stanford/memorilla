<h1 align="center">
  <img src="assets/gorilla.png" alt="" height="60" align="absmiddle" />
  &nbsp;Memorilla: Latent Semantic Memory for LLMs
</h1>

<p align="center">
  <a href="https://openreview.net/forum?id=VV0vvzL783"><img src="https://img.shields.io/badge/paper-OpenReview-blue?logo=arxiv&logoColor=white" alt="Paper"></a>
  <a href="https://memory.stanford.edu/"><img src="https://img.shields.io/badge/website-memory.stanford.edu-7E57C2?logo=googlechrome&logoColor=white" alt="Website"></a>
  <a href="https://huggingface.co/memorilla"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-memorilla-yellow" alt="Hugging Face"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0"></a>
</p>

This repository provides code for training and evaluating Memorilla, a memory system that gathers information from an
entire document collection into a fixed number of query-dependent memory tokens for a frozen LLM. The code is based on
our paper *[Memorilla: Latent Semantic Memory for LLMs](https://openreview.net/forum?id=VV0vvzL783)*.

A frozen LLM can only use what fits in its context window, so Memorilla reads an entire document collection, such as a
user's past chats or a long book, and hands the LLM just 16 memory tokens that carry what each question needs. With this
small budget, it achieves the best average performance against methods such as Mem0 and MemoRAG, and on FactKG it is
even more accurate than giving the LLM every document (84.33% vs. 80.43%) while using 108× fewer tokens. Furthermore, Memorilla can also help an agent keep learning from its own history, raising its average
success rate on TextWorld games from 18.6% to 24.2%.

<p align="center">
  <img src="assets/architecture.png" alt="Memorilla architecture" width="100%" />
</p>

## Contents

- [Installation](#installation)
- [Getting started](#getting-started)
- [Training and Evaluation](#training-and-evaluation)
- [Checkpoints](#checkpoints)
- [Using your own data](#using-your-own-data)
- [PersonalizationV4](#personalizationv4)
- [Reinforcement learning on TextWorld](#reinforcement-learning-on-textworld)
- [Documentation](#documentation)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

## Installation

Memorilla needs Linux with CUDA GPUs and Python 3.11; the recipes were run on NVIDIA H100 80GB GPUs. Install it with
[uv](https://docs.astral.sh/uv/):

```bash
$ git clone https://github.com/snap-stanford/memorilla.git && cd memorilla
$ uv venv --python 3.11 && source .venv/bin/activate
$ uv pip install -e .
$ uv pip install --no-build-isolation flash-attn==2.7.4.post1
```

<details>
<summary>FlashAttention-2 and optional extras</summary>

The last command installs FlashAttention-2, which training uses by default. Evaluation does not need it, and training
can use PyTorch's attention instead:

```bash
$ scripts/train.sh configs/stage1_enwiki.yaml --attn_implementation sdpa
```

Optional extras:

```bash
$ uv pip install -e ".[dev]"                 # tests, linters and pre-commit
$ uv pip install -e ".[wandb]"               # Weights & Biases logging
$ uv pip install -e ".[personalizationv4]"   # OpenAI client for the PersonalizationV4 pipeline
$ uv pip install -e ".[rl]"                  # TextWorld, for the RL data tools
```

</details>

## Getting started

Here is an example from the PersonaMem-v2 test set. The user has had 107 earlier conversations with an assistant,
about 32,000 tokens in total, and now asks a new question. Memorilla reads all of the conversations and hands them to
a frozen Qwen3-8B as just 16 memory tokens, about 2,000 times fewer. You need one 80 GB GPU, and the first run
downloads about 25 GB of models and the 37 MB test set.

```python
import torch
from huggingface_hub import snapshot_download

from memorilla import MemoryModule
from memorilla.benchmarks import PERSONA_PROMPT
from memorilla.data import apply_chat_template, build_messages, ensure_data, load_split
from memorilla.embeddings import embed_texts, load_encoder
from memorilla.vllm import MemoryVLLM

row = load_split("pmv2", "test", ensure_data("pmv2", ["test"]))[73]  # one user's question and past conversations
documents, question = row["documents"], row["question"]

encoder = load_encoder("Qwen/Qwen3-Embedding-4B", gpu_memory_utilization=0.25)
embeddings = torch.from_numpy(embed_texts(encoder, documents + [question]))  # one vector per text

checkpoint = snapshot_download("memorilla/Memorilla-Qwen3-8B", allow_patterns=["pmv2/*"])
model = MemoryVLLM("Qwen/Qwen3-8B", MemoryModule.from_pretrained(f"{checkpoint}/pmv2"))

messages = build_messages(question, PERSONA_PROMPT, num_memories=16)  # reserves 16 <|memory|> positions
prompt = apply_chat_template(model.tokenizer, messages, add_generation_prompt=True)
answer = model.generate(
    **model.tokenizer(prompt, return_tensors="pt"),
    doc_embeds=embeddings[None, :-1],
    doc_padding_mask=torch.zeros(1, len(documents), dtype=torch.bool),
    question_embeds=embeddings[-1:],
)
print(f"Question: {question}\nMemorilla: {answer[0].strip()}")
```

Output:

> **Question:** What are some easy breakfast ideas that will keep me full all morning?
>
> **Memorilla:** Since you’ve got a family history of type 2 diabetes, you might enjoy breakfasts that balance
> protein, healthy fats, and fiber to keep you full without spiking blood sugar—like scrambled eggs with avocado and a
> side of roasted veggies, or Greek yogurt with chia seeds and berries. These options give you steady energy while
> keeping your blood sugar in check.

> [!NOTE]
> The user never says that diabetes runs in their family. The only clue is in one of the 107 conversations, where they
> asked whether diabetes on the mother's side of the family raises a person's risk more than on the father's side.
> Memorilla connects that clue to a breakfast question that has nothing to do with it, while Qwen3-8B without the
> memory answers with a generic list of breakfasts.

This is just one example. Because the LLM always receives the same **16 memory tokens**, no matter how many documents
are stored, the approach can potentially scale to millions of tokens. In the paper, Memorilla compresses books and
movie scripts from NarrativeQA by **more than 4,000x**. On one H100, growing a collection from 1 to **512 documents
(about 128K tokens)** leaves Memorilla's response time at **0.74 seconds**, while putting all the documents in the
prompt slows the model down from 0.75 to **19.9 seconds, 27x slower**.

The other checkpoints listed under [Checkpoints](#checkpoints) load the same way.

## Training and Evaluation

The recipes below train the memory module from scratch, and evaluates it on benchmarks. The data and its precomputed embeddings are downloaded from
[`memorilla/Memorilla-Data`](https://huggingface.co/datasets/memorilla/Memorilla-Data) on first use.

```bash
# Train the three stages with the Qwen3-8B decoder on 4 GPUs.
$ scripts/train.sh configs/stage1_enwiki.yaml
$ scripts/train.sh configs/stage2_mixture.yaml
$ scripts/train.sh configs/stage3_multitask.yaml

# Continue on the personalisation benchmarks.
$ scripts/train.sh configs/personalization_pv4.yaml
$ scripts/train.sh configs/personalization_pmv2.yaml

# Evaluate a checkpoint on every benchmark (or name one, e.g. factkg).
$ scripts/evaluate.sh runs/stage3/epoch-04 all
$ scripts/evaluate.sh runs/personalization_pv4/epoch-04 pv4

# Text-only baselines: closed book, RAG over the top-5 documents, full context.
$ scripts/baselines.sh closed_book all
$ scripts/baselines.sh rag all --top_k 5
$ scripts/baselines.sh full_context triviaqa
```

On 4 H100 80GB GPUs, the three stages take about 19 hours and download about 310 GiB of data and embeddings, and
evaluating a checkpoint on all eight benchmarks takes about 15 minutes on one GPU. The [training](docs/training.md) and
[evaluation](docs/evaluation.md) guides describe the main options, and each script's `--help` lists all of them.

## Checkpoints

The trained memory modules are on Hugging Face at
[`memorilla/Memorilla-Qwen3-8B`](https://huggingface.co/memorilla/Memorilla-Qwen3-8B). All were trained with the
Qwen3-8B decoder and the Qwen3-Embedding-4B encoder, the defaults of `train.py` and `evaluate.py`.

| Checkpoint | Folder | Recipe |
| --- | --- | --- |
| Stage 1 | `stage1/` | `configs/stage1_enwiki.yaml` |
| Stage 2 | `stage2/` | `configs/stage2_mixture.yaml` |
| Stage 3 (`epoch-00`) | `stage3/` | `configs/stage3_multitask.yaml` |
| PersonalizationV4, from Stage 1 | `pv4/` | `configs/single_task.yaml` |
| PersonaMem-v2, from Stage 1 | `pmv2/` | `configs/single_task.yaml` |

```bash
$ hf download memorilla/Memorilla-Qwen3-8B --local-dir checkpoints
$ scripts/evaluate.sh checkpoints/pv4 pv4                                          # evaluate a checkpoint
$ scripts/train.sh configs/stage3_multitask.yaml --init_memory checkpoints/stage2  # continue from Stage 2
```

```python
from memorilla import MemoryModule

memory = MemoryModule.from_pretrained("checkpoints/stage3")
```

## Using your own data

Any collection of documents with questions and answers can be trained and evaluated without code changes. Write one
parquet file per split, named like `train-00000-of-00001.parquet`, under `my_data/data/my_task/`, with one row per
question and the columns `collection_id`, `question`, `answer` and `documents`, then pass the directory as the data
repository. The embeddings are computed on first use.

```bash
$ python train.py --config configs/single_task.yaml --data_repo my_data --datasets my_task --output_dir runs/my_task
$ python evaluate.py --data_repo my_data --config my_task --split test --scoring generation --checkpoint runs/my_task/epoch-04
```

The `single_task` recipe warm-starts from the Stage 2 checkpoint; pass `--init_memory ""` to train from scratch. The
[data guide](docs/data.md#using-your-own-data) covers the file layout, document chunking and scoring options.

## PersonalizationV4

PersonalizationV4 (PV4) is a synthetic personalisation benchmark released with Memorilla. Each of its 149 fictional
users has had 200 short conversations with an AI assistant, and each question places the user in a new scenario and
asks what they would most likely do or prefer. The questions are written to require combining at least two facts about
the user, which the conversations show but never state. The data is on
[Hugging Face](https://huggingface.co/datasets/memorilla/PersonalizationV4), and
[`personalizationv4/`](personalizationv4/README.md) documents the task, the data format and the generation pipeline.

## Reinforcement learning on TextWorld

Memorilla can also serve as the memory of an agent and learn from reward alone. [`rl/`](rl/README.md) trains the
memory module with GRPO on [TextWorld](https://github.com/microsoft/TextWorld) text adventures, using
[SkyRL-Memorilla](https://github.com/Adibvafa/SkyRL-Memorilla), a fork of [SkyRL](https://github.com/NovaSky-AI/SkyRL),
and documents its installation, data, training and evaluation.

<p align="center">
  <img src="assets/textworld_replay.gif" alt="Replay of a held-out TextWorld game with and without Memorilla" width="100%" />
</p>

A schematic replay of one held-out game (routes simplified, observations paraphrased). With only its full history in
context, the agent revisits explored rooms until the 50-turn limit; with 8 trained memory tokens added, it reaches the
goal.

## Documentation

| Guide | Contents |
| --- | --- |
| [Data](docs/data.md) | Released configs, row format, embedding layout, storage locations, using your own data |
| [Training](docs/training.md) | Options, recipes and stages, outputs and resuming, multiple GPUs and SLURM, single-task training |
| [Evaluation](docs/evaluation.md) | Benchmarks, metrics, outputs, and the closed-book, RAG and full-context baselines |
| [PersonalizationV4](personalizationv4/README.md) | Task and scoring, data format, generation pipeline, regeneration, license |
| [Reinforcement learning](rl/README.md) | TextWorld setup, installation, data, training and evaluation |
| [Troubleshooting](docs/troubleshooting.md) | Multi-GPU stalls, slow evaluation, offline machines |

## Development

```bash
$ uv pip install -e ".[dev]"
$ pytest                             # unit tests (CPU only, no network)
$ ruff check . && black --check .    # lint
$ pre-commit install                 # run the same checks on every commit
```

Set `MEMORILLA_TEST_CHECKPOINTS` to a directory of checkpoints to also check that each one loads strictly and runs.

## Citation

```bibtex
@inproceedings{fallahpour2026memorilla,
  title     = {Memorilla: Latent Semantic Memory for {LLM}s},
  author    = {Adibvafa Fallahpour and Parsa Idehpour and Vignesh Kothapalli and Nikita Mounier and Shirley Wu and Shayan Pardis and Jure Leskovec},
  booktitle = {COLM Workshop on Context Beyond the Window: Persistent Knowledge in Language Models},
  year      = {2026},
  url       = {https://openreview.net/forum?id=VV0vvzL783}
}
```

## License

The code is released under the [Apache 2.0 license](LICENSE). Each config of Memorilla-Data keeps the license of its
source dataset, and PersonalizationV4 is released under CC BY 4.0.
