#!/usr/bin/env bash
set -eo pipefail

source /home/jdiaz/miniconda3/etc/profile.d/conda.sh
conda activate base
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

SCOPE="${SCOPE:-pilot}"
STAGE="${STAGE:-generate}"
NUM_STEPS="${NUM_STEPS:-15000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/velocity_integration_benchmark}"

python scripts/synthetic_velocity_integration_benchmark.py \
  --scope "$SCOPE" \
  --stage "$STAGE" \
  --num_steps "$NUM_STEPS" \
  --output_root "$OUTPUT_ROOT" \
  "$@"
