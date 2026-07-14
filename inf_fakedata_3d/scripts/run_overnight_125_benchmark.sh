#!/usr/bin/env bash
set -eo pipefail

source /home/jdiaz/miniconda3/etc/profile.d/conda.sh
conda activate base
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/velocity_integration_benchmark_1_2_5min}"
NUM_STEPS="${NUM_STEPS:-15000}"

echo "Starting 1/2/5-minute velocity-integration benchmark"
echo "Output root: $OUTPUT_ROOT"
echo "Training steps: $NUM_STEPS"
echo "Design: 3 beam supports x 2 speeds x 3 integration products x 2 training modes = 36 runs"

echo "Stage 1/4: sequential training"
PYTHONUNBUFFERED=1 python scripts/synthetic_velocity_integration_benchmark.py \
  --scope overnight_125 \
  --stage train \
  --num_steps "$NUM_STEPS" \
  --output_root "$OUTPUT_ROOT"

echo "Stage 2/4: dense reconstruction analysis"
PYTHONUNBUFFERED=1 python scripts/synthetic_velocity_integration_benchmark.py \
  --scope overnight_125 \
  --stage analyze \
  --num_steps "$NUM_STEPS" \
  --output_root "$OUTPUT_ROOT"

echo "Stage 3/4: curvature checks"
PYTHONUNBUFFERED=1 python scripts/synthetic_velocity_integration_benchmark.py \
  --scope overnight_125 \
  --stage check \
  --num_steps "$NUM_STEPS" \
  --output_root "$OUTPUT_ROOT"

echo "Stage 4/4: event-level summary"
PYTHONUNBUFFERED=1 python scripts/summarize_velocity_integration_results.py \
  --root "$OUTPUT_ROOT"

echo "Benchmark and event-level summary complete."
