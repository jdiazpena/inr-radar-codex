# Scientific Source

This folder contains the implementation that performs data loading, field generation,
model construction, training, and reconstruction analysis.

| File | Responsibility |
|---|---|
| `models.py` | Neural-network components and the `MLPINR` model |
| `amisr_h5_reader_3d.py` | Reads AMISR HDF5 products into a physical-coordinate dataframe |
| `datasets.py` | Real radar HDF5 dataset loading and normalization |
| `synthetic_dataset.py` | Synthetic observation dataset and normalization |
| `synthetic_plasma.py` | Synthetic plasma fields, motion, beam geometries, and observation generation |
| `synthetic_train_3d.py` | Current synthetic trainer with derivative diagnostics and adaptive regularization |
| `synthetic_train_3d_window_reference_reg.py` | Earlier/reference synthetic training implementation |
| `train_radar_3d_window_reg.py` | Real-radar window trainer |
| `train_radar_3d_window_reference_reg_diagnostic.py` | Real-radar trainer with reference-ratio diagnostics |
| `synthetic_analyze_reconsturction_linear_errors.py` | Accepted reconstruction analysis in linear density units |
| `synthetic_analyze_reconsturction.py` | Older log-space analysis retained for comparison |

The next refactoring stage should split the large training and analysis files into
named modules and add function-level scientific explanations. This organizational
stage intentionally preserves their behavior and filenames.
