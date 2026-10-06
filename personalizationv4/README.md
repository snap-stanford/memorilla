# PersonalizationV4

PersonalizationV4 (PV4) is a synthetic personalisation benchmark released with Memorilla. Each user is a detailed
fictional persona who has had 200 short conversations with an AI assistant. The evaluation questions place the user in
a new scenario and ask what they would most likely do or prefer, and each one is written to require combining at least
two facts about the user. A model never sees the persona itself: it gets the user's conversations, in which those
traits are shown rather than stated, as its memory and answers in free text.

| | |
| --- | --- |
| Users | 149: 119 training users and 30 evaluation users, disjoint |
| Conversations per user | 200 two-turn chats (one user message, one assistant reply) |
| Questions per user | 133-145 (10 categories per persona) |
| Rows | train 15,058, validation 1,755, test 4,237 |
| Hard test subset | 629 questions |
| Scoring | nearest of five candidate answers under Qwen3-Embedding-4B |
| Data | [`memorilla/PersonalizationV4`](https://huggingface.co/datasets/memorilla/PersonalizationV4), and the `pv4` config of `memorilla/Memorilla-Data` |

## Task and scoring

Every question comes with five candidate answers: the reference answer and four distractors. The generation prompt
asks for distractors that a reasonable person might genuinely prefer and for candidates matched in length, grammatical
structure and specificity.

The model never sees the candidates. It reads the question (with the user's conversations available as memory) and
generates a free-text answer. The generation and the non-empty candidates are embedded with Qwen3-Embedding-4B; the
answer is correct when the candidate with the highest cosine similarity to the generation is the reference. Accuracy
is the mean over questions.

**Hard subset.** The `hard` column marks 629 test questions that the untrained Qwen3-4B-Instruct-2507 decoder gets
wrong in at least one of two settings: with no documents, or with only the single most relevant conversation
retrieved. `accuracy_hard` reports accuracy on these questions.

## Data

`memorilla/PersonalizationV4` has these columns (in the `pv4` config of `memorilla/Memorilla-Data`, `user_id` is named
`collection_id`):

| Column | Type | Description |
| --- | --- | --- |
| `user_id` | int64 | User the question is about; all rows of a user share the same `documents`. |
| `question` | string | Scenario-based question about the user. |
| `answer` | string | Reference answer (one of `choices`). |
| `choices` | list[string] | The five candidate answers in A-E order; a missing candidate is an empty string and is skipped by the scorer. |
| `documents` | list[string] | The user's 200 chats in topic order, each a `Leo: <user message>` turn followed by an `Assistant: <reply>` turn (`Leo` is a fixed speaker tag for the user). |
| `hard` | bool | True for questions in the hard subset (test split only). |

| Split | Users | Rows | Content |
| --- | --- | ---: | --- |
| `train` | 119 training users (ids 6-125, except 7) | 15,058 | each training user's questions minus a 10% held-out part |
| `validation` | the same 119 training users | 1,755 | the held-out 10% of each training user's questions |
| `test` | 30 evaluation users (ids 126-155) | 4,237 | every question of the evaluation users, 629 of them in the hard subset |

The evaluation users never appear in `train` or `validation`. The per-user hold-out draws a permutation of the user's
questions with `numpy.random.default_rng(23)` and holds out the first `ceil(0.1 * n)`; this is the rule of
`datasets.Dataset.train_test_split(test_size=0.1, seed=23)`. In the dataset repository, `data/user_splits.csv` lists
every user with its role (`training` or `evaluation`) and its number of questions in each split and in the hard
subset, and `raw/hard_subset.csv` identifies the hard questions by user id and question index.

```python
from datasets import load_dataset

pv4 = load_dataset("memorilla/PersonalizationV4")
row = pv4["test"][0]
print(row["question"], row["answer"], len(row["documents"]))
```

To train and evaluate on PV4, continue the Stage 3 checkpoint with `configs/personalization_pv4.yaml` and evaluate the
`pv4` benchmark. Memorilla's evaluation uses the system prompt "You are a personalized AI assistant. Answer the question
about the user based on your understanding of the user." and the baselines use "You are a helpful assistant."; both
decode greedily with up to 256 new tokens:

```bash
python train.py --config configs/personalization_pv4.yaml
python evaluate.py --benchmark pv4 --checkpoint runs/personalization_pv4/epoch-04
python evaluate_baselines.py --benchmark pv4 --method rag --top_k 5
```

## How the data was generated

Each user starts from a long-form persona profile (about 2,400 words on average) that expands a five-sentence seed
persona from [Synthetic-Persona-Chat](https://huggingface.co/datasets/google/Synthetic-Persona-Chat) into a detailed
life: identity, work, family and friends, hobbies and tastes, personality, daily routine and a secret project. The
profiles are released with the dataset (`raw/personas/`, with the seed persona of every user in
`raw/persona_seeds.csv`) and are the input of the pipeline in `personalizationv4/`.

| Step | Output (per user) | Model | Reasoning effort |
| --- | --- | --- | --- |
| 1. Question categories: the 10 most testable dimensions of the persona | `categories.txt` | `gpt-5.1` | `none` |
| 2. Chat topics: 200 one-sentence scenarios covering every facet of the persona, early sceptical and later reliant phases, and requests secretly related to hidden projects | `chat_topics.txt` | `gpt-5.1` | `none` |
| 3. Chats: one two-turn chat per topic; the user's traits are shown, never stated | `chats/<topic index>.txt` | `gpt-5-mini` | `minimal` |
| 4. Questions: 15 requested per category, each with a reference answer, four distractors and a rationale | `qa/<category index>.txt` | `gpt-5.1` | `none` |
| 5. Question table: parse the questions, drop malformed ones, and move each reference to a random letter with `random.Random(42)` | `qa.csv` | | |
| 6. Dataset: per-user splits, chats attached as documents, hard-subset flags | `{train,validation,test}.parquet`, `user_splits.csv` | | |

Steps 1-5 are `personalizationv4/generate.py`, step 6 is `personalizationv4/build_dataset.py` (which also writes
`user_splits.csv`), and the exact prompts are in `personalizationv4/prompts.py`. The question prompt asks for
scenario-embedded questions that require combining two or more persona facts, distractors that a reasonable person
might genuinely prefer (including the best practice this persona rejects), and candidates matched in length, structure
and specificity.

## Regenerating PV4

Run these commands from the repository root (the pipeline is the `personalizationv4` package, which `pip` does not
install). Rebuild the released parquet files from the released raw generation outputs, without API calls:

```bash
hf download memorilla/PersonalizationV4 --repo-type dataset --include "raw/*" --local-dir pv4_data
python -m personalizationv4.build_dataset \
    --users_dir pv4_data/raw/users \
    --hard_subset pv4_data/raw/hard_subset.csv \
    --output_dir pv4_data/data
```

Generate a new set of users from persona profiles (yours, or the released ones in `raw/personas/`). This calls the
OpenAI API; `uv pip install -e ".[personalizationv4]"` adds the OpenAI client:

```bash
export OPENAI_API_KEY=...
python -m personalizationv4.generate --persona_dir pv4_data/raw/personas --output_dir pv4_data/users
python -m personalizationv4.build_dataset --users_dir pv4_data/users --output_dir pv4_data/new
```

- Persona profiles are `user_N.txt` files; `N` is the user id. `--user_start` / `--user_end` restrict generation to a
  range of users, and the `build_dataset` options `--train_users` and `--test_users` (inclusive ranges, default
  `6-125` and `126-155`) choose which users go to train/validation and to test.
- Generation is resumable: every output is written atomically and existing files are never requested again, so an
  interrupted or partially failed run is completed by rerunning the same command.
- `--model`, `--chat_model`, the reasoning efforts, `--num_categories`, `--num_topics`, `--num_questions`,
  `--max_workers` and `--seed` are configurable; `python -m personalizationv4.generate --help` lists them.
- Model outputs are sampled, so a new run produces a new dataset. The hard subset is defined on the released questions;
  leave out `--hard_subset` for newly generated data.

To train or evaluate Memorilla on rebuilt or newly generated data, name the user column `collection_id`, place the files
in the data-repository layout (`<dir>/data/pv4/<split>-00000-of-00001.parquet`) and pass `--data_repo <dir>`; the
embeddings are computed on first use:

```bash
python -m personalizationv4.build_dataset --users_dir pv4_data/users --output_dir pv4_data/new --id_column collection_id
mkdir -p my_pv4/data/pv4
for split in train validation test; do
  mv pv4_data/new/$split.parquet my_pv4/data/pv4/$split-00000-of-00001.parquet
done
python train.py --config configs/personalization_pv4.yaml --data_repo my_pv4 --output_dir runs/my_pv4
python evaluate.py --benchmark pv4 --data_repo my_pv4 --checkpoint runs/my_pv4/epoch-04 --output_dir results/my_pv4
```

## License and attribution

- PV4 is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). All users are fictional; any
  resemblance to real people is coincidental.
- The seed personas come from Google's Synthetic-Persona-Chat (Jandaghi et al., 2023), released under CC BY 4.0.
- All text in PV4 (persona profiles, chat topics, chats, questions and answers) was generated with OpenAI models.

When using PV4, please cite Memorilla (see [Citation](../README.md#citation)) and Synthetic-Persona-Chat:

```bibtex
@article{jandaghi2023faithful,
  title   = {Faithful Persona-based Conversational Dataset Generation with Large Language Models},
  author  = {Jandaghi, Pegah and Sheng, XiangHai and Bai, Xinyi and Pujara, Jay and Sidahmed, Hakim},
  journal = {arXiv preprint arXiv:2312.10007},
  year    = {2023}
}
```
