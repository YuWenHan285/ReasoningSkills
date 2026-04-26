# Reasoning Skills Pipeline

<p align="center">
  <strong>A modular pipeline for building, indexing, and retrieving reusable reasoning skills with API-based language models.</strong>
</p>

<p align="center">
  Generate trajectories • Extract learned heuristics • Retrieve relevant skills • Inject them into prompts
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#project-structure">Project Structure</a> •
  <a href="#license">License</a>
</p>

---

## Features

- End-to-end workflow for skill-guided reasoning
- API-first design for hosted or OpenAI-compatible models
- Structured skill extraction with reusable heuristics and retrieval keywords
- Multiple retrieval backends:
  - BM25
  - Dense nearest-neighbor search
  - Hybrid retrieval
- Simple JSONL artifacts for easy inspection and debugging
- Lightweight, modular codebase with minimal setup

---

## Installation

```bash
conda create -n trs python=3.11 -y
conda activate trs
pip install -e .
```

---

## Configuration

Create your local environment file:

```bash
cp .env.example .env
```

Then fill in your API settings, for example:

```bash
OPENAI_API_KEY=...
OPENAI_API_BASE=http://your-endpoint/v1
```

---

## Quick Start

### 1. Generate trajectories

```bash
trs generate \
  --source deepmath \
  --dataset-name zwhe99/DeepMath-103K \
  --split train \
  --limit 100 \
  --model openai/gpt-4o-mini \
  --output-path runs/demo/01_generations.jsonl
```

### 2. Extract learned heuristics

```bash
trs extract \
  --input-path runs/demo/01_generations.jsonl \
  --model openai/gpt-4o \
  --output-path runs/demo/02_skills.jsonl
```

### 3. Build a retrieval index

```bash
trs build-index \
  --skills-path runs/demo/02_skills.jsonl \
  --backend bm25 \
  --output-dir runs/demo/index_bm25
```

### 4. Run inference with retrieved skills

```bash
trs infer \
  --input-path data/eval.jsonl \
  --question-field question \
  --backend bm25 \
  --index-dir runs/demo/index_bm25 \
  --model openai/gpt-4o-mini \
  --prompt-style normal \
  --top-k 1 \
  --output-path runs/demo/03_infer.jsonl
```

### Demo script

```bash
bash scripts/run_deepmath_demo.sh
```

---

## Workflow

The pipeline is organized into three stages:

1. **Trajectory Generation**  
   Generate reasoning traces from a dataset using a base model.

2. **Skill Extraction**  
   Convert trajectories into learned heuristics and retrieval keywords using a stronger model.

3. **Skill-Guided Inference**  
   Retrieve relevant skills for a new query and inject them into the model prompt.

---

## Project Structure

```text
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
│   ├── generation.py
│   ├── indexing.py
│   ├── inference.py
│   ├── prompts.py
│   ├── schemas.py
│   ├── datasets/
│   ├── retrieval/
│   └── utils/
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Retrieval Backends

- **BM25**: lexical retrieval for strong surface-form overlap
- **Dense**: embedding-based nearest-neighbor retrieval
- **Hybrid**: combined sparse and dense retrieval

---

## Prompt Styles

Prompt templates are defined in:

```text
src/trs/prompts.py
```

Available styles include:

- `normal`
- `only`
- `try_to`
- `short`
- `draft`

---

## Outputs

A typical run produces:

```text
runs/
└── demo/
    ├── 01_generations.jsonl
    ├── 02_skills.jsonl
    ├── index_bm25/
    └── 03_infer.jsonl
```

- `01_generations.jsonl`: raw model trajectories
- `02_skills.jsonl`: extracted learned heuristics and retrieval keywords
- `index_*`: retrieval artifacts
- `03_infer.jsonl`: retrieved skills and final model outputs

---

## Notes

- Start with a small subset before scaling up
- Keep learned heuristics short and reusable
- Use small `top-k` values to avoid prompt inflation
- Inspect retrieval quality early

---

## License

MIT License
