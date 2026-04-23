#!/usr/bin/env bash
set -euo pipefail

# Example usage:
# export OPENAI_API_KEY=...
# bash scripts/run_deepmath_demo.sh

trs end-to-end-math \
  --dataset-name zwhe99/DeepMath-103K \
  --train-split train \
  --test-split train \
  --train-start 0 \
  --train-limit 100 \
  --test-start 100 \
  --test-limit 20 \
  --base-model openai/gpt-4o-mini \
  --summarizer-model openai/gpt-4o \
  --inference-model openai/gpt-4o-mini \
  --prompt-style normal \
  --workdir runs/deepmath_demo
