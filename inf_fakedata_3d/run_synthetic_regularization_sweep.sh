#!/usr/bin/env bash
set -eo pipefail

source /home/jdiaz/miniconda3/etc/profile.d/conda.sh
conda activate base
set -u

SYNTHETIC_CSV="outputs/synthetic_high_amp_left_right/synthetic_observations.csv"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="synthetic_train_3d.py"

mkdir -p logs outputs

COMMON_ARGS=(
  --synthetic_csv "$SYNTHETIC_CSV"
  --window_start_index 0
  --window_size_records 31
  --epsilon_data 1e-6
  --lambda_warmup_steps 500
  --num_steps 15000
  --summary_every 500
  --component_grad_every 500
  --num_collocation 16384
  --num_diagnostic_collocation 16384
  --collocation_grid_nx 100
  --collocation_grid_ny 100
)

run_case() {
  local label="$1"
  local output_dir="$2"
  local log_path="$3"
  shift 3
  local extra_args=("$@")

  if [[ -f "$output_dir/model_final.pt" && -f "$output_dir/history.csv" ]]; then
    echo "[$label] complete output exists; skipping: $output_dir"
    return 0
  fi

  if [[ -d "$output_dir" ]]; then
    echo "[$label] output directory exists but is incomplete: $output_dir" >&2
    echo "Move it aside or set a new output directory before rerunning." >&2
    return 2
  fi

  echo "[$label] starting"
  echo "  output: $output_dir"
  echo "  log:    $log_path"

  "$PYTHON_BIN" "$TRAIN_SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --output_dir "$output_dir" \
    "${extra_args[@]}" \
    > "$log_path" 2>&1

  echo "[$label] finished"
}

run_case \
  "data_only" \
  "outputs/codex_sweep_high_amp_data_only_15000" \
  "logs/codex_sweep_high_amp_data_only_15000.log"

run_case \
  "xy030_t030" \
  "outputs/codex_sweep_high_amp_xy030_t030_15000" \
  "logs/codex_sweep_high_amp_xy030_t030_15000.log" \
  --reference_loss_weights \
  --target_xy_ratio 0.30 \
  --target_t_ratio 0.30

run_case \
  "xy070_t030" \
  "outputs/codex_sweep_high_amp_xy070_t030_15000" \
  "logs/codex_sweep_high_amp_xy070_t030_15000.log" \
  --reference_loss_weights \
  --target_xy_ratio 0.70 \
  --target_t_ratio 0.30

run_case \
  "xy070_t070" \
  "outputs/codex_sweep_high_amp_xy070_t070_15000" \
  "logs/codex_sweep_high_amp_xy070_t070_15000.log" \
  --reference_loss_weights \
  --target_xy_ratio 0.70 \
  --target_t_ratio 0.70
