# Reasoning Skills Pipeline

An end-to-end repository for building a skill-guided reasoning workflow with API-based language models.

This project provides three main stages:

1. Trajectory generation from datasets such as DeepMath or any local / Hugging Face dataset.
2. Skill extraction with a stronger model to produce structured Skill Cards and retrieval Keywords.
3. Inference-time retrieval with BM25, dense nearest-neighbor search, or hybrid retrieval, followed by prompt injection of the retrieved skill cards.

The implementation is designed so that, once you fill in your API configuration, you can run the full pipeline end to end.

---

## Design goals

This repository is built around a practical workflow for reusable reasoning:

- collect raw problem-solving trajectories offline
- distill them into short, reusable skill cards
- index those cards for retrieval
- retrieve the most relevant cards for a new problem
- inject them into the model prompt to encourage shorter, more directed reasoning

The defaults included here are only implementation choices, and you are expected to tune them for your own models, datasets, and latency / cost constraints.

---

## Repository layout

trs_repro/
├── configs/
│   ├── code_hybrid.yaml
│   └── deepmath_bm25.yaml
├── scripts/
│   └── run_deepmath_demo.sh
├── src/trs/
│   ├── cli.py
│   ├── clients.py
│   ├── config.py
│   ├── eval_math.py
│   ├── generation.py
│   ├── indexing.py
│   ├── inference.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── datasets/
│   │   ├── deepmath.py
│   │   └── generic.py
│   ├── retrieval/
│   │   ├── bm25.py
│   │   ├── dense.py
│   │   └── hybrid.py
│   └── utils/
│       ├── io.py
│       └── text.py
├── .env.example
├── pyproject.toml
└── README.md

---

## Installation

### 1. Create environment

conda create -n trs python=3.11 -y
conda activate trs

### 2. Install this repo

From the repo root:

pip install -e .

If you want optional stronger math verification support:

pip install -e .[math]

---

## API setup

This repo uses LiteLLM as the API layer instead of bare requests.

Why LiteLLM:

- one unified client for many providers
- OpenAI-style interface
- retries and consistent exceptions
- optional embeddings through the same abstraction
- easier migration across providers

Create .env from .env.example:

cp .env.example .env

Then fill in at least:

OPENAI_API_KEY=...

If you are using an OpenAI-compatible proxy / base URL, also set:

OPENAI_API_BASE=http://your-endpoint/v1

You can then pass LiteLLM model names such as:

- openai/gpt-4o-mini
- openai/gpt-4o
- anthropic/claude-...
- gemini/...
- any other provider supported by LiteLLM

If you want to use a separate model for embeddings, configure that in the YAML config or CLI flags.

---

## Supported data input modes

### A. DeepMath directly from Hugging Face

trs generate \
  --source deepmath \
  --dataset-name zwhe99/DeepMath-103K \
  --split train \
  --limit 100 \
  --model openai/gpt-4o-mini \
  --output-path runs/deepmath/01_generations.jsonl

### B. Generic Hugging Face dataset

trs generate \
  --source hf \
  --dataset-name your_dataset_name \
  --split train \
  --question-field question \
  --answer-field answer \
  --limit 100 \
  --model openai/gpt-4o-mini \
  --output-path runs/custom/01_generations.jsonl

### C. Local JSONL file

trs generate \
  --source local_jsonl \
  --input-path data/train.jsonl \
  --question-field question \
  --answer-field answer \
  --model openai/gpt-4o-mini \
  --output-path runs/local/01_generations.jsonl

Each input row should provide at least a question. If an answer is available, it will be stored and optionally used in evaluation or extraction.

---

## End-to-end workflow

The full workflow is:

1. Generate reasoning trajectories with a base model.
2. Extract reusable skills with a stronger model.
3. Build retrieval indices.
4. Run inference on new questions by retrieving relevant skills and injecting them into the prompt.

### Step 1. Generate trajectories

Example:

trs generate \
  --source deepmath \
  --dataset-name zwhe99/DeepMath-103K \
  --split train \
  --limit 500 \
  --model openai/gpt-4o-mini \
  --output-path runs/deepmath/01_generations.jsonl

This stage stores, for each example:

- question
- reference answer if available
- model output
- raw completion text
- optional parsed final answer
- token usage metadata if returned by the provider

---

## Trajectory format

The generation output is JSONL. A typical row looks like:

{
  "id": "deepmath_train_000001",
  "question": "...",
  "answer": "...",
  "model": "openai/gpt-4o-mini",
  "completion": "...",
  "final_answer": "...",
  "usage": {
    "prompt_tokens": 123,
    "completion_tokens": 456,
    "total_tokens": 579
  },
  "meta": {
    "dataset": "zwhe99/DeepMath-103K",
    "split": "train"
  }
}

---

## Step 2. Extract skill cards

Example:

trs extract \
  --input-path runs/deepmath/01_generations.jsonl \
  --model openai/gpt-4o \
  --output-path runs/deepmath/02_skills.jsonl

This stage converts raw trajectories into compact, structured skill cards and retrieval keywords.

The default skill schema is:

- Trigger
- Do
- Avoid
- Check
- Risk

Each skill entry also stores a keyword list for retrieval.

### Example skill card object

{
  "id": "skill_000001",
  "source_id": "deepmath_train_000001",
  "question": "...",
  "keywords": [
    "substitution",
    "chain rule",
    "composite function"
  ],
  "skill_card": {
    "trigger": [
      "integrand matches f'(x) * g(f(x)) pattern"
    ],
    "do": [
      "identify inner function",
      "substitute inner variable",
      "rewrite integral in simpler form"
    ],
    "avoid": [
      "trying unrelated identities first"
    ],
    "check": [
      "substitution derivative matches remaining factor"
    ],
    "risk": [
      "missing constants after substitution"
    ]
  }
}

---

## Skill extraction prompt behavior

The extraction stage is intentionally generic and avoids dataset-specific hard coding.

The extractor is instructed to:

- summarize the reusable reasoning pattern rather than the instance-specific answer
- avoid copying full solutions or long code
- produce short, actionable skill fields
- produce keywords that are useful for retrieval

You can customize the extraction prompt in:

src/trs/prompts.py

If you want stricter output control, modify the extraction schema in:

src/trs/schemas.py

---

## Step 3. Build indices

### BM25 index

trs build-index \
  --skills-path runs/deepmath/02_skills.jsonl \
  --backend bm25 \
  --output-dir runs/deepmath/index_bm25

### Dense index

trs build-index \
  --skills-path runs/deepmath/02_skills.jsonl \
  --backend dense \
  --embedding-model BAAI/bge-m3 \
  --output-dir runs/deepmath/index_dense

### Hybrid index

trs build-index \
  --skills-path runs/deepmath/02_skills.jsonl \
  --backend hybrid \
  --embedding-model BAAI/bge-m3 \
  --output-dir runs/deepmath/index_hybrid

The indexed retrieval key is built from the source question plus extracted keywords.

---

## Retrieval backends

### BM25

BM25 is useful when lexical overlap is strong. It is typically a good default for formula-heavy or terminology-heavy domains.

### Dense retrieval

Dense retrieval embeds the query and the indexed keys, then performs nearest-neighbor search.

### Hybrid retrieval

Hybrid retrieval combines BM25 candidates and dense candidates, then fuses scores and returns the final top-k results.

The fusion strategy is configurable in code. See:

src/trs/retrieval/hybrid.py

---

## Step 4. Inference with retrieved skills

Example:

trs infer \
  --input-path data/eval.jsonl \
  --question-field question \
  --backend bm25 \
  --index-dir runs/deepmath/index_bm25 \
  --model openai/gpt-4o-mini \
  --prompt-style normal \
  --top-k 1 \
  --output-path runs/deepmath/03_infer.jsonl

For hybrid retrieval:

trs infer \
  --input-path data/eval.jsonl \
  --question-field question \
  --backend hybrid \
  --index-dir runs/code/index_hybrid \
  --model openai/gpt-4o-mini \
  --prompt-style draft \
  --top-k 5 \
  --output-path runs/code/03_infer.jsonl

This stage:

1. reads each new question
2. retrieves top-k relevant skill cards
3. injects them into the prompt
4. calls the reasoning model
5. stores the response and metadata

---

## Prompt styles

Prompt templates are centralized in:

src/trs/prompts.py

Current styles include:

- normal
- only
- try_to
- short
- draft

You can select them with:

--prompt-style normal
--prompt-style only
--prompt-style try_to
--prompt-style short
--prompt-style draft

### Generic injected prompt structure

A typical injected prompt contains:

- a short system instruction
- an optional solving hints section
- the current problem
- optional budget or brevity instruction depending on the chosen prompt style

You can directly edit the templates if you want your own house style.

---

## Example: one-command math demo

A demo shell script is included:

bash scripts/run_deepmath_demo.sh

This script demonstrates:

1. generation on a small DeepMath subset
2. skill extraction
3. index building
4. inference on a held-out subset

Before running it, make sure:

- your environment is activated
- the repo is installed
- your .env file is filled in

---

## YAML configs

Two example configs are included:

configs/deepmath_bm25.yaml
configs/code_hybrid.yaml

These configs define common settings such as:

- provider model names
- retrieval backend
- top-k
- prompt style
- input/output paths
- optional embedding model
- generation parameters such as temperature and max tokens

You can either use the configs directly or override fields from the CLI.

---

## Output files

A typical run directory may contain:

runs/
└── deepmath/
    ├── 01_generations.jsonl
    ├── 02_skills.jsonl
    ├── index_bm25/
    │   ├── metadata.json
    │   ├── corpus.jsonl
    │   └── ...
    └── 03_infer.jsonl

### 01_generations.jsonl

Stores raw model trajectories and metadata.

### 02_skills.jsonl

Stores extracted skill cards and retrieval keywords.

### index_*/

Stores serialized retrieval artifacts.

### 03_infer.jsonl

Stores inference inputs, retrieved skills, final generations, and usage metadata.

---

## Evaluation

### Math

A simple evaluator is included in:

src/trs/eval_math.py

Depending on your task, you may want to compare:

- exact string match
- normalized answer match
- symbolic equivalence

The current implementation is intentionally lightweight and can be extended.

### Code

This repository includes retrieval and prompt injection support for coding tasks, but the exact judging environment should be customized to your benchmark.

You can plug in your own evaluator that:

- compiles generated code
- runs test cases
- computes pass@1 or any other desired metric

---

## DeepMath notes

When using DeepMath or similar math datasets, common practical choices are:

- use smaller subsets first for smoke testing
- cache intermediate JSONL files
- inspect extraction quality before indexing the entire corpus
- compare retrieval backends on a small validation split before scaling up

---

## Tips for prompt quality

Some practical tips:

- keep the skill cards short
- avoid including task-specific answers in cards
- do not inject too many skills at once unless your domain really benefits from it
- prefer top-k values that keep the context compact
- inspect false positives from retrieval early

In many cases, the quality of skill extraction matters more than adding more cards.

---

## Common commands

### Generate

trs generate \
  --source deepmath \
  --dataset-name zwhe99/DeepMath-103K \
  --split train \
  --limit 200 \
  --model openai/gpt-4o-mini \
  --output-path runs/demo/01_generations.jsonl

### Extract

trs extract \
  --input-path runs/demo/01_generations.jsonl \
  --model openai/gpt-4o \
  --output-path runs/demo/02_skills.jsonl

### Build BM25 index

trs build-index \
  --skills-path runs/demo/02_skills.jsonl \
  --backend bm25 \
  --output-dir runs/demo/index_bm25

### Infer

trs infer \
  --input-path data/eval.jsonl \
  --question-field question \
  --backend bm25 \
  --index-dir runs/demo/index_bm25 \
  --model openai/gpt-4o-mini \
  --prompt-style normal \
  --top-k 1 \
  --output-path runs/demo/03_infer.jsonl

---

## Troubleshooting

### 1. API calls fail immediately

Check:

- OPENAI_API_KEY or your provider key is set
- OPENAI_API_BASE is correct if using a proxy
- the LiteLLM model string matches your provider
- your provider supports the selected endpoint type

### 2. Dense retrieval build is slow

This is expected on large corpora. Try:

- a smaller subset first
- batching embeddings
- storing cached embeddings
- building on a machine with enough RAM

### 3. Retrieval quality is poor

Check:

- whether extracted keywords are too generic
- whether skill cards are too instance-specific
- whether top-k is too large
- whether BM25 or dense retrieval is more suitable for your domain

### 4. Prompt gets too long

Try:

- reducing top-k
- shortening each skill card
- using a more concise prompt template
- filtering low-quality cards before indexing

### 5. Final answers are verbose or unstable

Try:

- lowering temperature
- tightening the inference prompt
- adding a structured final-answer instruction
- post-processing the final answer for evaluation

---

## Extending the repository

This codebase is meant to be easy to modify.

Typical extensions include:

- plugging in a different embedding model
- adding reranking after retrieval
- adding a code execution judge
- adding symbolic math verification
- building domain-specific skill schemas
- adding caching or asynchronous API execution
- adding experiment tracking

---

## Minimal example workflow

A minimal small-scale run can look like this:

1. Generate 100 trajectories on a train split.
2. Extract 100 skill cards.
3. Build a BM25 index.
4. Run inference on 20 held-out examples.
5. Inspect retrieved skills and final outputs manually.
6. Iterate on extraction prompt and retrieval settings.
7. Scale up only after the small run looks reasonable.

---

## Notes on implementation philosophy
This repository intentionally favors:
- simple JSONL artifacts
- explicit stage boundaries
- inspectable intermediate outputs
- interchangeable retrieval backends
- provider-agnostic API calls
The goal is not to hide the pipeline behind a single opaque trainer, but to make every stage easy to inspect, replace, and debug.
---