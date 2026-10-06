# Data

All data lives in one public Hugging Face dataset repository,
[`memorilla/Memorilla-Data`](https://huggingface.co/datasets/memorilla/Memorilla-Data), with one config per dataset.
Scripts download only the configs and splits they use, together with their precomputed embeddings, on first use.

| Config | Contents | Splits |
| --- | --- | --- |
| `enwiki` | Wikipedia passages to restate (Stage 1) | train, validation |
| `squad_v2`, `drop`, `coqa`, `quail` | Reading comprehension (Stage 2) | train, validation |
| `pwc` | Instructions over a text: questions, summaries, extraction (Stage 2) | train, validation |
| `cnn_dailymail`, `samsum`, `dialogsum` | Summarisation of news, chats and dialogues (Stage 2) | train, validation |
| `msmarco` | Open-domain QA over retrieved web passages (Stage 2) | train, validation |
| `pubmedqa` | Biomedical yes/no/maybe questions over abstract sections (Stage 2 and benchmark) | train, test |
| `pmv2` | PersonaMem-v2: responses that fit a user's preferences, from their chat history | train, validation, test |
| `pv4` | [PersonalizationV4](../personalizationv4/README.md): scenario questions about a user, from their 200 conversations | train, validation, test |
| `narrativeqa` | Questions about books and movie scripts, over the full story | train, validation, test |
| `triviaqa` | Trivia questions over Wikipedia and web evidence | train, validation, test |
| `factkg` | Claim verification over knowledge-graph triples | train, validation, test |
| `lamp4`, `lamp7` | Personalised headline generation and tweet paraphrasing from a user's history | train, test |

Every row has:

| Field | Type | Meaning |
| --- | --- | --- |
| `collection_id` | int64 | Identifies the document collection (a user, a story, or the row itself). Rows that share it share `documents`. |
| `question` | string | The question or instruction. |
| `answer` | string | The reference answer. |
| `documents` | list[string] | The collection, one chunk per entry, in its original order. |
| `choices` | list[string] | `pmv2`, `pv4`, `factkg` only: answer candidates for scoring (`answer` is one of them). |
| `hard` | bool | `pv4` only: the hard test subset. |

Each split also ships Qwen3-Embedding-4B vectors of every document and question:

```
data/<config>/<split>-XXXXX-of-YYYYY.parquet
embeddings/qwen3-embedding-4b/<config>/<split>/
    index.safetensors            collection_ids, shard, start, length
    documents-XXXXX.safetensors  float16 document embeddings, whole collections per shard
    questions.safetensors        float16 question embeddings, row-aligned with the parquet files
```

Locations are controlled by environment variables, and every script accepts the matching flag:

| Variable | Flag | Default | Meaning |
| --- | --- | --- | --- |
| `MEMORILLA_HOME` | | current directory | Root for relative paths such as `runs/`, `results/` and `data/` |
| `MEMORILLA_DATA_REPO` | `--data_repo` | `memorilla/Memorilla-Data` | Hub dataset repo, or a local directory with the same layout |
| `MEMORILLA_DATA_DIR` | `--data_dir` | `$MEMORILLA_HOME/data` | Where a Hub repo is mirrored locally (unused when the data repo is a local directory) |
| `MEMORILLA_MODEL_CACHE` | `--cache_dir` | Hugging Face default | Cache for the encoder and decoder weights |

The whole repository is about 320 GiB, almost all of it embeddings. A full training run downloads about 27 GiB for
Stage 1, 60 GiB for Stage 2 and 223 GiB for Stage 3; `evaluate.py --benchmark all` needs about 11 GiB. Each config
keeps the license of its source dataset; the [dataset card](https://huggingface.co/datasets/memorilla/Memorilla-Data)
lists every config's size, source and license.

## Using your own data

Any collection of documents with questions and answers can be trained and evaluated without code changes.

**1. Write parquet files** in the data-repository layout, one folder per config:

```python
from pathlib import Path

import pandas as pd

rows = [
    {
        "collection_id": 0,
        "question": "Which city did the user move to last spring?",
        "answer": "Lisbon",
        "documents": ["User: We finally signed the lease in Lisbon ...", "User: Packing up the Berlin flat ..."],
    },
    # one row per question; rows that share a collection_id share the same documents
]
folder = Path("my_data/data/my_task")
folder.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_parquet(folder / "train-00000-of-00001.parquet")
# likewise validation-00000-of-00001.parquet and test-00000-of-00001.parquet
```

- `collection_id` is an integer; rows that share it must have identical `documents`, since each collection is
  embedded once, from its first row. Every collection needs at least one document.
- Each entry of `documents` is embedded into one vector, so split long texts into passages; the benchmarks use chunks
  of roughly 10 to a few hundred tokens (single triples, tweets, story paragraphs, chat sessions).
- Training needs a `train` split; a `validation` split is optional and is validated on when present.
- Add a `choices` column (a list of candidates that contains `answer`) to score by answer choice.

**2. Train.** Pass the directory as the data repository. On first use, the embeddings of every needed split are
computed with the encoder (vLLM, one GPU) and written to `my_data/embeddings/qwen3-embedding-4b/my_task/<split>/`:

```bash
python train.py --config configs/single_task.yaml --data_repo my_data --datasets my_task --output_dir runs/my_task
```

The `single_task` recipe warm-starts from the Stage 2 checkpoint (`runs/stage2/epoch-00`); pass `--init_memory ""` to
train from scratch. To train on your data together with released configs, download those into the same directory first
and list them all in `--datasets`:

```bash
hf download memorilla/Memorilla-Data --repo-type dataset --local-dir my_data \
    --include "data/pv4/*" "embeddings/qwen3-embedding-4b/pv4/*"
python train.py --config configs/single_task.yaml --data_repo my_data --datasets my_task pv4 --output_dir runs/my_mix
```

**3. Evaluate** any split with one scoring method (`generation`, `exact_match`, `polarity` or `choice`):

```bash
python evaluate.py --data_repo my_data --config my_task --split test --scoring generation --checkpoint runs/my_task/epoch-04
```

Unless overridden, this uses the training system prompt ("You are a helpful assistant."), a 2,048-token prompt limit
and 128 new tokens. The baselines run on the registered benchmarks only.
