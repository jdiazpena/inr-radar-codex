# Tests

The current tests are lightweight checks rather than a complete scientific test suite.

| Test | Checks |
|---|---|
| `test_velocity_integration_benchmark.py` | The overnight benchmark has exactly the intended 18 cases and 36 runs |
| `test_synthetic_plasma.py` | Synthetic generation produces valid observations, truth, and geometry |
| `test_synthetic_dataset.py` | Synthetic normalization, denormalization, and prediction-frame construction |

Run from the project root with the Conda base environment active.
