# Synthetic Velocity and Integration-Time Benchmark

This benchmark tests whether the current INR reconstructs moving plasma structure
consistently as beam support, plasma speed, and ISR integration time change. It
also includes a separate flow-reversal velocity-shear stress case.

## Observation model

Each synthetic record is the trapezoidal average of linear electron density over
a centered integration window. The result is converted to `log10(Ne)` only after
integration. The CSV retains both the integration-averaged target and the
instantaneous truth at the window midpoint.

The analytical field is defined outside the nominal event interval so that the
first and last centered exposures remain valid. This is intentional and is noted
in the generated configuration.

## Scopes

- `smoke`: one ordinary 23-beam case plus one flow-reversal case.
- `pilot`: 42 and 23 beams, 0.36 and 2.0 km/s, 1 and 10 minute integrations,
  plus one 23-beam flow-reversal case.
- `core`: 42, 23, and 11 beams; 0.36, 1.0, and 2.0 km/s; 1, 2, 5, and 10
  minute integrations; three seeds; plus three flow-reversal seeds.
- `full`: the core matrix plus selected 45 and 90 degree direction tests.

The 11-beam cases are stress tests. A trustworthy outcome may be a poor
reconstruction accompanied later by an appropriately low reliability score.

## Stages

The wrapper activates the Conda base environment itself:

```bash
cd /home/jdiaz/postdoc/codex-inr-radar/inf_fakedata_3d
SCOPE=pilot STAGE=generate bash scripts/run_velocity_integration_benchmark.sh
SCOPE=pilot STAGE=train bash scripts/run_velocity_integration_benchmark.sh
SCOPE=pilot STAGE=analyze bash scripts/run_velocity_integration_benchmark.sh
SCOPE=pilot STAGE=check bash scripts/run_velocity_integration_benchmark.sh
```

Use `STAGE=all` only when a complete scope should run sequentially. Existing
complete outputs are skipped. An incomplete training directory is restarted from
a fresh model in the same directory.

For a quick pipeline test:

```bash
SCOPE=smoke STAGE=all NUM_STEPS=20 bash scripts/run_velocity_integration_benchmark.sh --cpu
```

## Regularization and curvature checks

Every data case is trained as both `data_only` and the current `xy030_t030`
baseline. This first benchmark is not the final regularization search.

`STAGE=check` writes `curvature_health.csv`. It summarizes the final quarter of
the training history and flags spatial collapse only if all spatial second
derivatives are effectively zero; temporal collapse is assessed separately from
`ftt`. The flags are diagnostic rather than automatic grounds for rejecting a
case, because a heavily time-averaged target can be physically close to flat.

## Outputs

- `benchmark_manifest.csv`: one row per training run and all factor settings.
- `data/<case_id>/`: observations, selected dense truth, geometry, and config.
- `runs/<case_id>/<regularization>/`: model, history, diagnostics, and analysis.
- `benchmark_metrics.csv`: integrated-target and midpoint-target error metrics.
- `curvature_health.csv`: second-derivative non-collapse diagnostics.

The primary reconstruction score is against the integration-averaged truth,
which is what the synthetic radar observed. Midpoint error is reported alongside
it to quantify temporal smearing and the loss of instantaneous-state fidelity.
