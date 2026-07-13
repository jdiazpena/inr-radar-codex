# CODEX Synthetic Sweep Report

## Synthetic Test

This experiment trains an INR/SIREN-like coordinate network on sparse synthetic radar-like samples from a known analytic plasma field. The field is a high-amplitude moving Gaussian-like electron-density enhancement on a 500 km by 500 km domain, evaluated over a 31-record time window.

Synthetic data are useful here because the true electron density and true gradients are known everywhere on the evaluation grid, not only at sparse radar/sample points. That makes it possible to measure dense reconstruction error, gradient recovery, and regularization behavior under controlled conditions.

## Point Types

- Radar/sample points: sparse observed synthetic measurements used in the data loss.
- Collocation points: x-y-time points used to compute curvature regularization terms during training.
- Diagnostic collocation points: fixed probe points used to log second-derivative and gradient-norm diagnostics.
- Plotting/evaluation grid: dense x-y grids used after training to compare predictions against analytic truth.

## Four Runs

| Run | Output directory | Regularization |
| --- | --- | --- |
| data_only | `outputs/codex_sweep_high_amp_data_only_15000` | data loss only |
| xy030_t030 | `outputs/codex_sweep_high_amp_xy030_t030_15000` | reference ratios xy=0.30, t=0.30 |
| xy070_t030 | `outputs/codex_sweep_high_amp_xy070_t030_15000` | reference ratios xy=0.70, t=0.30 |
| xy070_t070 | `outputs/codex_sweep_high_amp_xy070_t070_15000` | reference ratios xy=0.70, t=0.70 |

## Training Settings

All runs used the same synthetic CSV, window, model defaults, collocation settings, diagnostics, and 15000 training steps. The common data source was `outputs/synthetic_high_amp_left_right/synthetic_observations.csv`.

Common settings included window start 0, 31 records, `epsilon_data=1e-6`, `lambda_warmup_steps=500`, `num_collocation=16384`, `num_diagnostic_collocation=16384`, and a 100 by 100 collocation grid.

## Diagnostic Check

The data-only run now keeps diagnostic collocation active even though curvature terms are not included in the loss. Its `history.csv` contains populated `fxx`, `fxy`, `fyy`, `ftt`, `grad_norm_total`, `grad_norm_data`, `grad_norm_xy_weighted`, and `grad_norm_t_weighted` columns. The weighted curvature gradient norms are correctly zero for the data-only run.

## Error Metrics

The analysis uses `synthetic_analyze_reconsturction_linear_errors.py`, so density errors are computed in physical linear density units after converting predicted `log10(Ne)` back to `Ne`.

Metrics include density RMSE, MAE, bias, p95 absolute error, and max absolute error. Gradient metrics include dx, dy, dt, combined gradient magnitude RMSE, and p95 gradient absolute error.

Regional metrics were computed for:

- Full domain.
- Interior domain with a 10 percent boundary strip removed.
- Near-observation region within 50 km of a synthetic sample point at that time.
- High-gradient region, top 25 percent by true combined gradient magnitude.

## Main Results

The baseline regularized run `xy030_t030` is the strongest overall choice in this sweep.

| Region | Best density RMSE | Best gradient RMSE |
| --- | --- | --- |
| Full domain | `xy030_t030`, 2.6305e10 m^-3 | `xy030_t030`, 3.2015e-03 |
| Interior | `xy030_t030`, 1.2626e10 m^-3 | `xy030_t030`, 2.8129e-03 |
| Near observation | `xy030_t030`, 2.2981e10 m^-3 | `xy030_t030`, 3.1428e-03 |
| High gradient | `data_only`, 3.8654e10 m^-3 | `xy030_t030`, 4.4843e-03 |

Interpretation:

1. The data-only run fits the sparse observations very strongly, but has larger derivative diagnostics and worse gradient recovery than baseline. This supports the concern that unconstrained fits can become more oscillatory between points.
2. The baseline `0.30/0.30` regularization reduces artifacts enough to improve full-domain, interior, near-observation, and gradient metrics without obviously smearing the moving structure.
3. Stronger spatial regularization `0.70/0.30` did not improve gradient recovery in this run. It slightly worsened full-domain density and combined gradient RMSE relative to baseline.
4. Stronger spatial and temporal regularization `0.70/0.70` was not beneficial here. It had the worst full-domain density RMSE and did not beat baseline on gradient metrics, suggesting oversmoothing or overconstraint for this synthetic case.
5. Best overall density reconstruction: `xy030_t030`, except in the high-gradient-only density region where `data_only` has lower density RMSE.
6. Best gradient reconstruction: `xy030_t030` in every region.
7. The best density run and best gradient run are the same for the full domain, interior, and near-observation regions. They differ in the high-gradient region, where data-only wins density RMSE but baseline wins gradient RMSE.

## Figures And Tables

Generated comparison files:

- `comparison/regional_error_summary.csv`
- `comparison/regularization_sweep_summary.csv`
- `comparison/training_final_rows.csv`
- `comparison/density_rmse_by_region.png`
- `comparison/density_p95_by_region.png`
- `comparison/gradient_rmse_by_region.png`
- `comparison/loss_history_comparison.png`
- `comparison/derivative_diagnostic_comparison.png`
- `comparison/regularization_sweep_findings.md`

Each run also has an `error_analysis/` directory with dense reconstruction CSVs, per-time summaries, regional summaries, and truth/prediction/error figures.

## Notes

The 23-point synthetic geometry was added as `--sample_geometry sparse_23`. Existing 42-point geometries and defaults were preserved.
