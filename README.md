# Reasoning Skills Pipeline

A lightweight pipeline for building and using reusable reasoning skills with API-based language models.

This repo supports three stages:

1. Generate reasoning trajectories from a dataset
2. Extract structured skill cards and keywords with a stronger model
3. Retrieve relevant skills at inference time with BM25 / dense / hybrid search

## Install

conda create -n trs python=3.11 -y
conda activate trs
pip install -e .

## API setup

Copy:

cp .env.example .env

Then fill in your API settings, for example:

OPENAI_API_KEY=...
OPENAI_API_BASE=http://your-endpoint/v1

## Main commands

Generate trajectories:

trs generate \
  --source deepmath \
  --dataset-name zwhe99/DeepMath-103K \
  --split train \
  --limit 100 \
  --model openai/gpt-4o-mini \
  --output-path runs/demo/01_generations.jsonl

Extract skill cards:

trs extract \
  --input-path runs/demo/01_generations.jsonl \
  --model openai/gpt-4o \
  --output-path runs/demo/02_skills.jsonl

Build BM25 index:

trs build-index \
  --skills-path runs/demo/02_skills.jsonl \
  --backend bm25 \
  --output-dir runs/demo/index_bm25

Run inference with retrieved skills:

trs infer \
  --input-path data/eval.jsonl \
  --question-field question \
  --backend bm25 \
  --index-dir runs/demo/index_bm25 \
  --model openai/gpt-4o-mini \
  --prompt-style normal \
  --top-k 1 \
  --output-path runs/demo/03_infer.jsonl

## Prompt styles

Available prompt styles:

- normal
- only
- try_to
- short
- draft

Prompt templates are defined in:

src/trs/prompts.py

## Retrieval backends

- bm25
- dense
- hybrid

## Output files

01_generations.jsonl   raw model trajectories
02_skills.jsonl        extracted skill cards and keywords
index_*/               retrieval index files
03_infer.jsonl         retrieved skills and final model outputs

## Example

Run the demo script:

bash scripts/run_deepmath_demo.sh

## Notes

- Start with a small subset first
- Keep skill cards short
- Use small top-k values to avoid prompt inflation
- Inspect retrieval quality before scaling up

## License

Add your preferred license here.