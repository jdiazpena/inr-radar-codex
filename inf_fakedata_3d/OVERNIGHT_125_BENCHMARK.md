# Overnight 1/2/5-Minute Benchmark

This benchmark is the full ordinary-motion factorial requested after the pilot:

| Factor | Values |
|---|---|
| Beam support | 42, 23, 11 beams |
| Horizontal speed | 0.36, 2.00 km/s |
| Integration product | 1, 2, 5 minutes |
| Training mode | data only, `xy030_t030` |
| Seed | 0 |
| Training steps | 15,000 by default |

The design contains 18 synthetic data cases and 36 trained models. Each model is
analyzed with `synthetic_analyze_reconsturction_linear_errors.py`. Reconstruction
metrics use the integration-averaged synthetic field corresponding to the radar
measurement product. Instantaneous midpoint metrics are retained only as contextual
diagnostics and must not be used to rank models.

The rare flow-reversal case is intentionally excluded from this factorial. It remains
a separate morphology stress test and should not be mixed into estimates of ordinary
beam, speed, or integration-product effects.

## Run

From WSL:

```bash
cd /home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d
bash run_overnight_125_benchmark.sh
```

The launcher activates `/home/jdiaz/miniconda3` automatically. It generates data,
trains all models, runs dense linear-error analysis, checks curvature health, and
writes the event-level comparison tables. Training output is connected directly to
the terminal so the original live progress display is preserved. Output is isolated
under:

```text
outputs/velocity_integration_benchmark_1_2_5min/
```

Completed data and model runs are skipped when the command is launched again. A run
interrupted while its current model directory is incomplete is restarted from a fresh
model in that directory. Training is sequential; no cases run in parallel.

## Monitor

Count completed models from another WSL terminal:

```bash
find /home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d/outputs/velocity_integration_benchmark_1_2_5min/runs \
  -name model_final.pt | wc -l
```

Expected final count: `36`.

## Optional Overrides

Use a shorter run only for verification:

```bash
NUM_STEPS=10 OUTPUT_ROOT=outputs/velocity_integration_benchmark_1_2_5min_smoke \
  bash run_overnight_125_benchmark.sh
```

The scientific overnight run should use the default 15,000 steps.
