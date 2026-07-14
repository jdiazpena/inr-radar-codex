# Experiment Scripts

This folder contains entry points that coordinate the scientific code in `src/`.
They should be run from the project root, although the shell launchers resolve the
root automatically.

| Script | Purpose |
|---|---|
| `run_overnight_125_benchmark.sh` | Sequential 1/2/5-minute, 42/23/11-beam benchmark |
| `run_velocity_integration_benchmark.sh` | General velocity/integration benchmark launcher |
| `synthetic_velocity_integration_benchmark.py` | Defines cases and coordinates generation, training, analysis, and checks |
| `summarize_velocity_integration_results.py` | Produces event-level physical reconstruction summaries |
| `run_synthetic_regularization_sweep.sh` | Four-case regularization sweep |
| `analyze_synthetic_sweep.py` | Dense analysis for the regularization sweep |
| `compare_synthetic_regularization_sweep.py` | Comparison tables and plots for the sweep |
| `run_reference_windows.py` | Repeated real-radar window training |
| `run_reference_windows_diagnostics_minimal.py` | Repeated real-radar windows with derivative diagnostics |

Shell launchers activate `/home/jdiaz/miniconda3` before enabling strict Bash variable
checking. Python workflows run one case at a time unless their documentation says
otherwise.
