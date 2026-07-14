# INR Radar Reconstruction

This project reconstructs continuous ionospheric plasma fields from sparse radar
measurements using implicit neural representations (INRs). It contains workflows for
both synthetic experiments and real radar windows.

The repository is organized by responsibility so that source code, executable
workflows, documentation, tests, and generated results are not mixed together.

## Project Map

| Path | Purpose |
|---|---|
| `src/` | Scientific implementation: models, datasets, synthetic generation, training, and reconstruction analysis |
| `scripts/` | Commands that coordinate experiments, parameter sweeps, benchmarks, and repeated radar windows |
| `config/` | Home for configuration files as hard-coded experiment settings are externalized |
| `tests/` | Fast structural tests and synthetic smoke tests |
| `docs/` | Research plans, benchmark protocols, findings, and runbooks |
| `outputs/` | Generated datasets, checkpoints, histories, figures, tables, and archived results |

Raw radar data are not copied into this repository. Real-data workflows receive the
HDF5 input path as a command-line argument or through their current defaults.

## Main Workflows

Run commands from the project root:

```bash
cd /home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d
```

### Overnight 1/2/5-minute synthetic benchmark

```bash
bash scripts/run_overnight_125_benchmark.sh
```

This runs 42-, 23-, and 11-beam cases at 0.36 and 2.00 km/s for 1-, 2-, and
5-minute integration products. The data-only and `xy030_t030` models are trained
sequentially, giving 36 total model runs.

### Original velocity/integration benchmark

```bash
SCOPE=pilot STAGE=all bash scripts/run_velocity_integration_benchmark.sh
```

### Synthetic regularization sweep

```bash
bash scripts/run_synthetic_regularization_sweep.sh
```

### Tests

```bash
source /home/jdiaz/miniconda3/etc/profile.d/conda.sh
conda activate base
python -m unittest -q tests/test_velocity_integration_benchmark.py
python tests/test_synthetic_plasma.py
python tests/test_synthetic_dataset.py
```

The two synthetic smoke tests write into `outputs/synthetic_smoke_test/`.

## Reading Order

For a first pass through the code:

1. Read `docs/CODEX_INR_PLAN.md` for the scientific context and current decisions.
2. Read `src/models.py` for the SIREN-style INR definition.
3. Read `src/synthetic_plasma.py` and `src/synthetic_dataset.py` for the synthetic field and observations.
4. Read `src/synthetic_train_3d.py` for the training and adaptive regularization workflow.
5. Read `src/synthetic_analyze_reconsturction_linear_errors.py` for the accepted physical-unit reconstruction analysis.
6. Read `scripts/synthetic_velocity_integration_benchmark.py` to see how complete experiments are assembled.
7. For real radar, read `src/amisr_h5_reader_3d.py` followed by `src/datasets.py`.

The spelling `reconsturction` is retained temporarily because existing workflows use
that filename. Renaming and internal readability improvements belong to the next code
refactoring stage.

## Output Policy

`outputs/` is generated state, not source code. Do not use files there as hidden
configuration inputs unless a workflow explicitly documents that dependency.
Root-level artifacts from earlier work were preserved in
`outputs/archive_root_analysis/`; regularization comparison products live in
`outputs/comparison/`, and historical sweep logs live in `outputs/logs/`.
