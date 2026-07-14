# -*- coding: utf-8 -*-
"""
amisr_h5_reader_3d.py

Read AMISR HDF5 data and extract a 3D INR training dataframe.

3D here means:
    x, y, time

not:
    x, y, z

The altitude is fixed by selecting one altitude band, usually the same
band chosen during the 2D experiment.

This file does NOT:
    - normalize data
    - create torch tensors
    - train the model
    - save a cache file

It only reads the HDF5 once and returns a pandas DataFrame.

Main output columns:
    x_km
    y_km
    t_sec
    log10_Ne

Useful metadata columns:
    time_index
    unix_start
    unix_end
    unix_mid
    t_hours
    beam_index
    beamcode
    az_deg
    el_deg
    range_index
    range_km
    altitude_km
    z_km
    Ne
    dNe
    rel_dNe
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


# ============================================================
# Small helpers
# ============================================================

def _as_array(dataset) -> np.ndarray:
    """
    Convert an HDF5 dataset into a numpy array.
    """

    return np.asarray(dataset[()])


def _to_km(values: np.ndarray, name: str) -> np.ndarray:
    """
    Convert an array to km if it appears to be in meters.

    AMISR files commonly store range/altitude in meters.
    Some derived files may already be in km.

    Heuristic:
        median absolute value > 2000 means meters
    """

    values = np.asarray(values, dtype=np.float64)

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError(f"{name}: no finite values found.")

    median_abs = float(np.nanmedian(np.abs(finite)))

    if median_abs > 2000.0:
        return values / 1000.0

    return values


def _parse_utc_to_unix(value: str | float | int | None) -> float | None:
    """
    Parse a UTC time selector.

    Accepted:
        None
        unix seconds as int/float
        "YYYY-mm-dd HH:MM:SS"
        "YYYY-mm-ddTHH:MM:SS"
        strings with trailing Z
    """

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()

    if text == "":
        return None

    # Numeric string, unix seconds.
    try:
        return float(text)
    except ValueError:
        pass

    text = text.replace("Z", "+00:00")

    # datetime.fromisoformat accepts both "T" and "+00:00".
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Fallback for "YYYY-mm-dd HH:MM:SS".
        dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return float(dt.timestamp())


# def _select_time_indices(
#     unix_time: np.ndarray,
#     time_start_utc: str | float | int | None,
#     time_end_utc: str | float | int | None,
#     record_stride: int,
#     max_records: int | None,
# ) -> np.ndarray:
#     """
#     Select AMISR time records using midpoint time.
#     """

#     unix_time = np.asarray(unix_time, dtype=np.float64)

#     if unix_time.ndim != 2 or unix_time.shape[1] != 2:
#         raise ValueError(
#             f"Time/UnixTime must have shape [Ntime, 2], got {unix_time.shape}"
#         )

#     unix_mid = 0.5 * (unix_time[:, 0] + unix_time[:, 1])

#     start_unix = _parse_utc_to_unix(time_start_utc)
#     end_unix = _parse_utc_to_unix(time_end_utc)

#     mask = np.ones(unix_mid.shape, dtype=bool)

#     if start_unix is not None:
#         mask &= unix_mid >= start_unix

#     if end_unix is not None:
#         mask &= unix_mid <= end_unix

#     indices = np.where(mask)[0]

#     if record_stride <= 0:
#         raise ValueError("record_stride must be >= 1")

#     indices = indices[::record_stride]

#     if max_records is not None:
#         if max_records <= 0:
#             raise ValueError("max_records must be positive when provided.")

#         indices = indices[:max_records]

#     if indices.size == 0:
#         raise ValueError("No time records selected.")

#     return indices.astype(int)

def _select_time_indices(
    unix_time: np.ndarray,
    time_start_utc: str | float | int | None,
    time_end_utc: str | float | int | None,
    record_stride: int,
    max_records: int | None,
    window_start_index: int | None = None,
    window_size_records: int | None = None,
) -> np.ndarray:
    """
    Select AMISR time records.

    Two modes:

    1. Time/UTC mode:
        use time_start_utc, time_end_utc, record_stride, max_records

    2. Window-index mode:
        use window_start_index and window_size_records

        Example:
            window_start_index = 100
            window_size_records = 11

        selects original HDF5 records:
            100, 101, ..., 110

    record_stride is still applied after the initial selection.
    max_records is also applied at the end if provided.
    """

    unix_time = np.asarray(unix_time, dtype=np.float64)

    if unix_time.ndim != 2 or unix_time.shape[1] != 2:
        raise ValueError(
            f"Time/UnixTime must have shape [Ntime, 2], got {unix_time.shape}"
        )

    if record_stride <= 0:
        raise ValueError("record_stride must be >= 1")

    n_times = unix_time.shape[0]
    unix_mid = 0.5 * (unix_time[:, 0] + unix_time[:, 1])

    # ------------------------------------------------------------
    # Mode 1: explicit HDF5 record window
    # ------------------------------------------------------------
    if window_start_index is not None or window_size_records is not None:
        if window_start_index is None or window_size_records is None:
            raise ValueError(
                "window_start_index and window_size_records must be provided together."
            )

        window_start_index = int(window_start_index)
        window_size_records = int(window_size_records)

        if window_start_index < 0:
            raise ValueError("window_start_index must be >= 0.")

        if window_size_records <= 0:
            raise ValueError("window_size_records must be > 0.")

        window_end_index = window_start_index + window_size_records

        if window_start_index >= n_times:
            raise ValueError(
                f"window_start_index={window_start_index} is outside "
                f"the file with n_times={n_times}."
            )

        if window_end_index > n_times:
            raise ValueError(
                f"Requested window [{window_start_index}, {window_end_index}) "
                f"extends beyond n_times={n_times}."
            )

        indices = np.arange(window_start_index, window_end_index, dtype=int)

    # ------------------------------------------------------------
    # Mode 2: UTC/time selection, old behavior
    # ------------------------------------------------------------
    else:
        start_unix = _parse_utc_to_unix(time_start_utc)
        end_unix = _parse_utc_to_unix(time_end_utc)

        mask = np.ones(unix_mid.shape, dtype=bool)

        if start_unix is not None:
            mask &= unix_mid >= start_unix

        if end_unix is not None:
            mask &= unix_mid <= end_unix

        indices = np.where(mask)[0]

    # ------------------------------------------------------------
    # Apply stride and optional cap
    # ------------------------------------------------------------
    indices = indices[::record_stride]

    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max_records must be positive when provided.")

        indices = indices[:max_records]

    if indices.size == 0:
        raise ValueError("No time records selected.")

    return indices.astype(int)


def _read_beamcodes(f: h5py.File) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read beamcode, azimuth, and elevation.

    Current supported format:
        /BeamCodes with columns:
            beamcode, azimuth_deg, elevation_deg
    """

    if "BeamCodes" not in f:
        raise KeyError(
            "This reader currently expects /BeamCodes in the HDF5 file."
        )

    beamcodes = _as_array(f["BeamCodes"])

    if beamcodes.ndim != 2 or beamcodes.shape[1] < 3:
        raise ValueError(
            f"/BeamCodes must have shape [Nbeams, >=3], got {beamcodes.shape}"
        )

    beamcode = beamcodes[:, 0].astype(float)
    az_deg = beamcodes[:, 1].astype(float)
    el_deg = beamcodes[:, 2].astype(float)

    return beamcode, az_deg, el_deg


def _force_beam_range_shape(
    arr: np.ndarray,
    n_beams: int,
    n_ranges: int,
    name: str,
) -> np.ndarray:
    """
    Force an array to shape [Nbeams, Nranges].

    Some files may store [Nbeams, Nranges].
    If [Nranges, Nbeams], transpose.
    """

    arr = np.asarray(arr)

    if arr.shape == (n_beams, n_ranges):
        return arr

    if arr.shape == (n_ranges, n_beams):
        return arr.T

    raise ValueError(
        f"{name} has shape {arr.shape}, expected "
        f"({n_beams}, {n_ranges}) or ({n_ranges}, {n_beams})."
    )


def _compute_radar_xyz_from_range(
    range_km: np.ndarray,
    az_deg: np.ndarray,
    el_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute local radar-centered x/y/z coordinates from range, azimuth, elevation.

    Convention:
        x_km: east-like
        y_km: north-like
        z_km: up-like

    Formula:
        x = r cos(el) sin(az)
        y = r cos(el) cos(az)
        z = r sin(el)

    Angles are degrees.
    """

    range_km = np.asarray(range_km, dtype=np.float64)
    az_rad = np.deg2rad(np.asarray(az_deg, dtype=np.float64))
    el_rad = np.deg2rad(np.asarray(el_deg, dtype=np.float64))

    n_beams, n_ranges = range_km.shape

    az2 = az_rad[:, None]
    el2 = el_rad[:, None]

    x_km = range_km * np.cos(el2) * np.sin(az2)
    y_km = range_km * np.cos(el2) * np.cos(az2)
    z_km = range_km * np.sin(el2)

    if x_km.shape != (n_beams, n_ranges):
        raise RuntimeError("Internal coordinate shape error.")

    return x_km, y_km, z_km


def _select_range_gate_per_beam(
    altitude_km: np.ndarray,
    h0_km: float,
    half_width_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For each beam, select one range gate.

    Selection rule:
        among gates inside [h0-half_width, h0+half_width],
        choose the gate closest to h0.

    Returns:
        beam_indices
        range_indices
    """

    if half_width_km <= 0:
        raise ValueError("half_width_km must be > 0")

    h_min = h0_km - half_width_km
    h_max = h0_km + half_width_km

    beam_indices = []
    range_indices = []

    n_beams = altitude_km.shape[0]

    for beam_idx in range(n_beams):
        alt = altitude_km[beam_idx, :]

        in_band = (
            np.isfinite(alt)
            & (alt >= h_min)
            & (alt <= h_max)
        )

        candidate_ranges = np.where(in_band)[0]

        if candidate_ranges.size == 0:
            continue

        best_local = np.argmin(np.abs(alt[candidate_ranges] - h0_km))
        range_idx = int(candidate_ranges[best_local])

        beam_indices.append(beam_idx)
        range_indices.append(range_idx)

    if len(beam_indices) == 0:
        raise ValueError(
            "No beam/range gates found inside the requested altitude band."
        )

    return np.asarray(beam_indices, dtype=int), np.asarray(range_indices, dtype=int)


# ============================================================
# Main reader
# ============================================================

def read_amisr_h5_3d_altitude_band(
    h5_path: str | Path,
    h0_km: float,
    half_width_km: float,
    time_start_utc: str | float | int | None = None,
    time_end_utc: str | float | int | None = None,
    record_stride: int = 1,
    max_records: int | None = None,
    window_start_index: int | None = None,
    window_size_records: int | None = None,
    min_ne: float = 0.0,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Read AMISR HDF5 and build a 3D dataframe for INR training.

    The output is one selected altitude-band point per beam per time record.

    Coordinates:
        x_km, y_km, t_sec

    Target:
        log10_Ne

    h0_km:
        center altitude of the fixed band

    half_width_km:
        altitude half-width around h0_km

    time_start_utc / time_end_utc:
        optional time limits.
        Can be unix seconds or UTC strings.

    record_stride:
        use every Nth selected time record.

    max_records:
        optional cap on number of time records.

    min_ne:
        minimum allowed Ne. Keep 0.0 for "Ne must be positive".
    """

    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    rows = []

    with h5py.File(h5_path, "r") as f:
        if "FittedParams" not in f:
            raise KeyError("HDF5 file is missing /FittedParams.")

        required = [
            "FittedParams/Ne",
            "FittedParams/Range",
            "FittedParams/Altitude",
            "Time/UnixTime",
        ]

        for key in required:
            if key not in f:
                raise KeyError(f"HDF5 file is missing required dataset: {key}")

        beamcode, az_deg, el_deg = _read_beamcodes(f)

        ne_all = _as_array(f["FittedParams"]["Ne"]).astype(np.float64)

        if ne_all.ndim != 3:
            raise ValueError(
                f"FittedParams/Ne must have shape [Ntime, Nbeams, Nranges], "
                f"got {ne_all.shape}"
            )

        n_times, n_beams, n_ranges = ne_all.shape

        if len(beamcode) != n_beams:
            raise ValueError(
                f"BeamCodes has {len(beamcode)} beams but Ne has {n_beams} beams."
            )

        if "dNe" in f["FittedParams"]:
            dne_all = _as_array(f["FittedParams"]["dNe"]).astype(np.float64)

            if dne_all.shape != ne_all.shape:
                raise ValueError(
                    f"FittedParams/dNe shape {dne_all.shape} does not match "
                    f"Ne shape {ne_all.shape}."
                )
        else:
            dne_all = None

        range_km = _to_km(
            _as_array(f["FittedParams"]["Range"]),
            name="FittedParams/Range",
        )

        altitude_km = _to_km(
            _as_array(f["FittedParams"]["Altitude"]),
            name="FittedParams/Altitude",
        )

        range_km = _force_beam_range_shape(
            range_km,
            n_beams=n_beams,
            n_ranges=n_ranges,
            name="FittedParams/Range",
        )

        altitude_km = _force_beam_range_shape(
            altitude_km,
            n_beams=n_beams,
            n_ranges=n_ranges,
            name="FittedParams/Altitude",
        )

        unix_time = _as_array(f["Time"]["UnixTime"]).astype(np.float64)

        time_indices = _select_time_indices(
            unix_time=unix_time,
            time_start_utc=time_start_utc,
            time_end_utc=time_end_utc,
            record_stride=record_stride,
            max_records=max_records,
            window_start_index=window_start_index,
            window_size_records=window_size_records,
)

        unix_mid_all = 0.5 * (unix_time[:, 0] + unix_time[:, 1])
        t0 = float(unix_mid_all[time_indices[0]])

        beam_indices, range_indices = _select_range_gate_per_beam(
            altitude_km=altitude_km,
            h0_km=float(h0_km),
            half_width_km=float(half_width_km),
        )

        x_all, y_all, z_all = _compute_radar_xyz_from_range(
            range_km=range_km,
            az_deg=az_deg,
            el_deg=el_deg,
        )

        if verbose:
            print("AMISR H5 3D reader")
            print(f"  h5_path:          {h5_path}")
            print(f"  Ne shape:         {ne_all.shape}")
            print(f"  selected times:   {len(time_indices)}")
            print(f"  selected beams:   {len(beam_indices)}")
            print(f"  h0_km:            {h0_km}")
            print(f"  half_width_km:    {half_width_km}")
            print(f"  record_stride:    {record_stride}")
            print(f"  max_records:      {max_records}")
            print(f"  window_start:     {window_start_index}")
            print(f"  window_size:      {window_size_records}")

        for time_idx in time_indices:
            ne_t = ne_all[time_idx, :, :]

            ne_sel = ne_t[beam_indices, range_indices]

            valid = np.isfinite(ne_sel) & (ne_sel > min_ne)

            if dne_all is not None:
                dne_t = dne_all[time_idx, :, :]
                dne_sel = dne_t[beam_indices, range_indices]
            else:
                dne_sel = np.full(ne_sel.shape, np.nan, dtype=np.float64)

            # Do not require dNe to be finite for now.
            selected_beams = beam_indices[valid]
            selected_ranges = range_indices[valid]
            selected_ne = ne_sel[valid]
            selected_dne = dne_sel[valid]

            if selected_ne.size == 0:
                continue

            unix_start = float(unix_time[time_idx, 0])
            unix_end = float(unix_time[time_idx, 1])
            unix_mid = float(unix_mid_all[time_idx])
            t_sec = unix_mid - t0

            log10_ne = np.log10(selected_ne)

            rel_dne = np.full(selected_ne.shape, np.nan, dtype=np.float64)
            good_dne = np.isfinite(selected_dne) & (selected_ne > 0.0)
            rel_dne[good_dne] = selected_dne[good_dne] / selected_ne[good_dne]

            n = selected_ne.size

            block = {
                "time_index": np.full(n, int(time_idx), dtype=int),
                "unix_start": np.full(n, unix_start, dtype=np.float64),
                "unix_end": np.full(n, unix_end, dtype=np.float64),
                "unix_mid": np.full(n, unix_mid, dtype=np.float64),
                "t_sec": np.full(n, t_sec, dtype=np.float64),
                "t_hours": np.full(n, t_sec / 3600.0, dtype=np.float64),

                "beam_index": selected_beams.astype(int),
                "beamcode": beamcode[selected_beams].astype(float),
                "az_deg": az_deg[selected_beams].astype(float),
                "el_deg": el_deg[selected_beams].astype(float),
                "range_index": selected_ranges.astype(int),

                "range_km": range_km[selected_beams, selected_ranges],
                "altitude_km": altitude_km[selected_beams, selected_ranges],
                "x_km": x_all[selected_beams, selected_ranges],
                "y_km": y_all[selected_beams, selected_ranges],
                "z_km": z_all[selected_beams, selected_ranges],

                "Ne": selected_ne.astype(np.float64),
                "dNe": selected_dne.astype(np.float64),
                "rel_dNe": rel_dne.astype(np.float64),
                "log10_Ne": log10_ne.astype(np.float64),
            }

            rows.append(pd.DataFrame(block))

    if len(rows) == 0:
        raise ValueError("No valid AMISR samples were extracted.")

    df = pd.concat(rows, ignore_index=True)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["x_km", "y_km", "t_sec", "log10_Ne"]).copy()
    df = df.sort_values(["time_index", "beam_index"]).reset_index(drop=True)

    if verbose:
        print()
        print("Extracted dataframe:")
        print(f"  rows:             {len(df)}")
        print(f"  time records:     {df['time_index'].nunique()}")
        print(f"  x range [km]:     {df['x_km'].min():.3f} to {df['x_km'].max():.3f}")
        print(f"  y range [km]:     {df['y_km'].min():.3f} to {df['y_km'].max():.3f}")
        print(f"  t range [s]:      {df['t_sec'].min():.3f} to {df['t_sec'].max():.3f}")
        print(f"  log10_Ne range:   {df['log10_Ne'].min():.6f} to {df['log10_Ne'].max():.6f}")

    return df