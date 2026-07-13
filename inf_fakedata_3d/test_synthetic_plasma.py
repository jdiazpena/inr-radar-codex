# -*- coding: utf-8 -*-
"""
test_synthetic_plasma.py

Small smoke test for synthetic_plasma.py.

Run:
    python3 test_synthetic_plasma.py

Expected output:
    outputs/synthetic_smoke_test/
        synthetic_observations.csv
        synthetic_truth_selected_times.csv
        synthetic_sample_geometry.csv
        synthetic_config.json
        truth_samples_time_0000.png
        truth_samples_time_0015.png
        truth_samples_time_0030.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from synthetic_plasma import build_parser, generate_synthetic_case


def main() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--output_dir", "outputs/synthetic_smoke_test",
            "--domain_size_km", "500",
            "--duration_sec", "3600",
            "--n_times", "31",
            "--motion", "left_right",
            "--sample_geometry", "sparse_grid",
            "--truth_grid_nx", "101",
            "--truth_grid_ny", "101",
            "--seed", "0",
        ]
    )

    paths = generate_synthetic_case(args)

    obs = pd.read_csv(paths["observations_csv"])
    truth = pd.read_csv(paths["truth_selected_csv"])
    geom = pd.read_csv(paths["geometry_csv"])

    assert len(obs) == 31 * 42, f"Unexpected obs rows: {len(obs)}"
    assert obs["time_index"].nunique() == 31
    assert obs["beam_id"].nunique() == 42
    assert len(geom) == 42
    assert truth["time_index"].nunique() == 3
    assert obs["log10_Ne"].min() > 0.0
    assert obs["Ne"].min() > 0.0

    print("Synthetic smoke test passed")
    print(f"  output_dir: {paths['output_dir']}")
    print(f"  obs rows:   {len(obs)}")
    print(f"  truth rows: {len(truth)}")
    print(f"  geometry:   {len(geom)} locations")


if __name__ == "__main__":
    main()
