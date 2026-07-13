# CODEX INR Synthetic Sweep Plan

Working copy inspected: `/home/jdiaz/postdoc/codex_inr_radar`

Active project folder: `/home/jdiaz/postdoc/codex_inr_radar/inf_fakedata_3d`

Note: on this machine `/home/jdiaz/postdoc` resolves through a symlink to
`/mnt/wsl/data/home-storage/jdiaz/postdoc`. The user confirmed this is expected.
Codex commands should explicitly activate conda before running Python:

```bash
source /home/jdiaz/miniconda3/etc/profile.d/conda.sh
conda activate base
```

## 1. Synthetic Data Generation

The main synthetic data generator is `synthetic_plasma.py`.

Responsibilities:

- Defines the analytic synthetic plasma field as `log10(Ne)` from a moving
  anisotropic Gaussian enhancement on a uniform background.
- Provides analytical first derivatives of `log10(Ne)` with respect to
  physical coordinates: `x_km`, `y_km`, and `t_sec`.
- Generates sparse observation CSVs, selected dense truth grids, geometry CSVs,
  configuration JSON, and truth/sample plots.
- Current geometry modes are:
  - `sparse_grid`: default 7 by 6 layout, 42 points.
  - `fan`: default 6 ranges by 7 angles, 42 points.
  - `random`: default 42 random points.

Important current output for this task:

- `outputs/synthetic_high_amp_left_right/synthetic_observations.csv`
- `outputs/synthetic_high_amp_left_right/synthetic_config.json`
- `outputs/synthetic_high_amp_left_right/synthetic_sample_geometry.csv`
- `outputs/synthetic_high_amp_left_right/synthetic_truth_selected_times.csv`
- `outputs/synthetic_high_amp_left_right/truth_samples_time_0000.png`
- `outputs/synthetic_high_amp_left_right/truth_samples_time_0015.png`
- `outputs/synthetic_high_amp_left_right/truth_samples_time_0030.png`

The requested 23-beam configuration is not currently present in
`synthetic_plasma.py`. I propose adding it later as a new explicit geometry mode,
without changing the existing 42-point defaults.

## 2. Synthetic INR Training

The main synthetic trainer is `synthetic_train_3d.py`.

Responsibilities:

- Reads synthetic observations through `SyntheticPlasmaTimeDataset` from
  `synthetic_dataset.py`.
- Trains an `MLPINR` model from `models.py` with coordinates
  `[x_norm, y_norm, t_norm]` and target normalized `log10(Ne)`.
- Supports full-batch or minibatch data fitting.
- Supports optional second-derivative curvature regularization:
  - spatial curvature: mean of `fxx^2 + 2 fxy^2 + fyy^2`
  - temporal curvature: mean of `ftt^2`
- Supports fixed lambda mode and reference-ratio lambda mode.
- Saves `history.csv`, `run_config.json`, final and best checkpoints,
  measured-point predictions, training plots, derivative diagnostics, gradient
  norm diagnostics, and x-y prediction plots.

Related real-data trainers inspected:

- `/home/jdiaz/postdoc/codex_inr_radar/inf_amisr_3d/train_radar_3d_window_reference_reg.py`
- `/home/jdiaz/postdoc/codex_inr_radar/inf_amisr_3d/train_radar_3d_window_reference_reg_diagnostic.py`
- `inf_fakedata_3d/train_radar_3d_window_reference_reg_diagnostic.py`

These mirror the same AMISR-style x/y/time INR structure and reference-ratio
curvature logic, but use `RadarTimeH5Dataset` from `datasets.py` rather than the
synthetic CSV dataset.

## 3. Reconstruction Analysis

The single-run analysis script to use for future work is:

- `synthetic_analyze_reconsturction_linear_errors.py`

The filename keeps the existing misspelling `reconsturction`, but this is the
preferred reconstruction analysis file because it handles the linear-density
errors properly. The older `synthetic_analyze_reconsturction.py` also exists,
but should not be the primary basis for the sweep analysis.

Responsibilities:

- Loads one trained model checkpoint.
- Reconstructs the analytical synthetic truth from `synthetic_config.json`.
- Evaluates the model on a dense x-y grid at selected times.
- Writes dense reconstruction CSVs and summary CSVs.
- Plots truth, prediction, and error.
- Optionally computes first-gradient errors using autograd and the chain rule
  from normalized coordinates back to physical units.

Current dense analysis outputs exist under:

- `outputs/synthetic_train_high_amp_win0_diag_1500/error_analysis/`
- `outputs/synthetic_train_high_amp_win0_diag_5000/error_analysis/`

I have not yet modified or refactored either analysis script.

## 4. Important Command-Line Controls

### Data and Window

`synthetic_train_3d.py` uses:

- `--synthetic_csv`: synthetic observations CSV.
- `--target_col`: target column, default `log10_Ne`.
- `--window_start_index`: first time-record index in the training window.
- `--window_size_records`: number of time records in the training window.

### Model

- `--activation`: `relu`, `tanh`, `softplus`, or `sine`.
- `--hidden_features`
- `--hidden_layers`
- `--first_omega_0`
- `--hidden_omega_0`

### Data Loss and Training

- `--lr`
- `--batch_size`: `0` means full batch.
- `--num_steps`
- `--seed`
- `--cpu`

The data loss is MSE between predicted and observed normalized `log10(Ne)` at
synthetic radar/sample points.

### Regularization

Fixed-lambda mode:

- `--lambda_curv_xy`
- `--lambda_curv_t`

Reference-ratio mode:

- `--reference_loss_weights`
- `--target_xy_ratio`
- `--target_t_ratio`
- `--epsilon_data`
- `--loss_ema_beta`
- `--curvature_ema_floor`
- `--lambda_smoothing`
- `--lambda_update_every`
- `--lambda_warmup_steps`
- `--freeze_lambdas_after_step`
- `--lambda_curv_xy_min`
- `--lambda_curv_xy_max`
- `--lambda_curv_t_min`
- `--lambda_curv_t_max`

Regularization ramp:

- `--reg_ramp_frac`

### Collocation and Diagnostics

Training curvature collocation:

- `--num_collocation`
- `--collocation_grid_nx`
- `--collocation_grid_ny`
- `--grid_padding_frac`
- `--nearest_radius_factor`

Derivative diagnostics:

- `--num_diagnostic_collocation`
- `--deriv_zero_epsilon`

Gradient norm diagnostics:

- `--component_grad_every`

Logging:

- `--log_every`
- `--summary_every`
- `--disable_tqdm`
- `--resume_history`

Plotting/output:

- `--output_dir`
- `--grid_nx`
- `--grid_ny`
- `--grid_chunk_size`
- `--num_plot_times`
- `--no_plots`
- `--save_grid_csv`

Current observation: derivative diagnostic columns exist in `history.csv`, but
`synthetic_train_3d.py` only creates collocation and diagnostic collocation
points when `use_xy_curv` or `use_t_curv` is true. Therefore, a pure data-only
run will likely write `NaN` derivative diagnostics unless a minimal diagnostic
decoupling patch is added later.

## 5. Existing Outputs

Synthetic data cases:

- `outputs/synthetic_smoke_test/`
- `outputs/synthetic_left_right_test/`
- `outputs/synthetic_high_amp_left_right/`

Training runs:

- `outputs/synthetic_train_win0_test/`
- `outputs/synthetic_train_high_amp_win0_diag_1500/`
- `outputs/synthetic_train_high_amp_win0_diag_5000/`

The directory name `synthetic_train_high_amp_win0_diag_1500` is misleading:
the request says this was actually a 15000-step reference run. Future output
directories should use names ending in `15000` when they use 15000 steps.

Typical existing run artifacts include:

- `history.csv`
- `run_config.json`
- `model_final.pt`
- `model_best_total_after_ramp.pt`
- `model_best_data_after_ramp.pt`
- `predictions_at_measured_points.csv`
- `training_history.png`
- `derivative_rms_diagnostics.png`
- `derivative_near_zero_fraction.png`
- `gradient_norm_diagnostics.png`
- `xy_time_index_*.png`
- `error_analysis/dense_reconstruction_time_*.csv`
- `error_analysis/error_summary_by_time.csv`
- `error_analysis/error_summary_mean.csv`
- `error_analysis/truth_pred_error_time_*.png`
- `error_analysis/gradient_error_time_*.png`

No `logs/`, `comparison/`, or `codex_sweep_*` output directories were present
at this inspection step.

## 6. Files Proposed To Add

After approval, I propose adding:

- `run_synthetic_regularization_sweep.sh`
- `analyze_synthetic_sweep.py`
- `compare_synthetic_regularization_sweep.py`
- `CODEX_SYNTHETIC_SWEEP_REPORT.md`
- `comparison/regularization_sweep_summary.csv`
- `comparison/regional_error_summary.csv`
- comparison figures produced by the comparison script

I also propose adding one small synthetic geometry extension:

- a 23-point/23-beam geometry option in `synthetic_plasma.py`, preserving all
  existing geometry modes and defaults.

Potential minimal patch, only if confirmed by the data-only run:

- decouple derivative diagnostic collocation from regularization activation in
  `synthetic_train_3d.py`, so diagnostics can remain populated even when the
  training loss is data-only.

## 7. Files Proposed Not To Touch

Do not modify the original project:

- `/home/jdiaz/postdoc/inr-radar`

Inside the safe copy, avoid touching these unless a later approved goal truly
requires it:

- `models.py`: model definition is already adequate for the planned sweep.
- `datasets.py`: real AMISR dataset logic should remain unchanged.
- `/home/jdiaz/postdoc/codex_inr_radar/inf_amisr_3d/train_radar_3d_window_reference_reg.py`
- `/home/jdiaz/postdoc/codex_inr_radar/inf_amisr_3d/train_radar_3d_window_reference_reg_diagnostic.py`
- existing output directories under `outputs/`

For the sweep itself, prefer new scripts and new output directories. Avoid
rewriting `synthetic_train_3d.py` unless the data-only diagnostics check proves
that a small diagnostic-only patch is necessary.
