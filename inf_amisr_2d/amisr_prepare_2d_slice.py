# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd


def read_amisr_h5(filepath):
    """
    Read AMISR fitted HDF5 file.

    Expected structure:
        BeamCodes
        FittedParams/Ne
        FittedParams/dNe
        FittedParams/Range
        FittedParams/Altitude
        Time/UnixTime
    """

    filepath = Path(filepath)
    data = {}

    with h5py.File(filepath, "r") as f:
        if "BeamCodes" not in f:
            raise KeyError("Missing BeamCodes in HDF5 file.")

        beamcodes = np.asarray(f["BeamCodes"][()])
        if beamcodes.ndim != 2 or beamcodes.shape[1] < 3:
            raise ValueError(f"BeamCodes has unexpected shape: {beamcodes.shape}")

        data["beamcodes"] = beamcodes[:, 0]
        data["az"] = beamcodes[:, 1].astype(float)
        data["el"] = beamcodes[:, 2].astype(float)

        data["Ne"] = np.asarray(f["FittedParams"]["Ne"], dtype=float)

        if "dNe" in f["FittedParams"]:
            data["dNe"] = np.asarray(f["FittedParams"]["dNe"], dtype=float)
        else:
            data["dNe"] = None

        data["range"] = np.asarray(f["FittedParams"]["Range"], dtype=float)
        data["altitude"] = np.asarray(f["FittedParams"]["Altitude"], dtype=float)

        unix_time = np.asarray(f["Time"]["UnixTime"], dtype=float)
        data["unix_time"] = unix_time
        data["unix_mid"] = 0.5 * (unix_time[:, 0] + unix_time[:, 1])

    return data


def print_summary(data):
    print("Loaded AMISR data:")
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:12s}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key:12s}: {value}")


def to_km(arr, units="auto"):
    arr = np.asarray(arr, dtype=float)

    if units == "km":
        return arr

    if units == "m":
        return arr / 1000.0

    if units != "auto":
        raise ValueError("units must be auto, m, or km")

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise ValueError("Array has no finite values.")

    median_value = np.nanmedian(finite)

    if median_value > 2000.0:
        return arr / 1000.0

    return arr


def get_2d_geometry(arr, time_index, name):
    """
    Return a [beam, range] geometry array.

    Supports:
        [beam, range]
        [time, beam, range]
    """

    arr = np.asarray(arr, dtype=float)

    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        return arr[time_index, :, :]

    raise ValueError(f"{name} must be 2D or 3D, got shape {arr.shape}")


def make_xyz_for_time(data, time_index, range_units="auto", altitude_units="auto"):
    """
    Compute radar-centered x/y/z coordinates in km for one time index.

    x_km > 0 east
    y_km > 0 north
    z_km > 0 up from radar
    """

    range_2d = get_2d_geometry(data["range"], time_index, "range")
    altitude_2d = get_2d_geometry(data["altitude"], time_index, "altitude")

    range_km = to_km(range_2d, range_units)
    altitude_km = to_km(altitude_2d, altitude_units)

    az = np.deg2rad(np.asarray(data["az"], dtype=float))[:, None]
    el = np.deg2rad(np.asarray(data["el"], dtype=float))[:, None]

    if range_km.shape[0] != az.shape[0]:
        raise ValueError(
            f"Range has {range_km.shape[0]} beams, but az/el has {az.shape[0]} beams."
        )

    x_km = range_km * np.cos(el) * np.sin(az)
    y_km = range_km * np.cos(el) * np.cos(az)
    z_km = range_km * np.sin(el)

    return x_km, y_km, z_km, range_km, altitude_km


def convex_hull_area_km2(points_xy):
    pts = np.asarray(points_xy, dtype=float)
    pts = pts[np.all(np.isfinite(pts), axis=1)]

    if pts.shape[0] < 3:
        return 0.0

    pts = np.unique(pts, axis=0)

    if pts.shape[0] < 3:
        return 0.0

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)

    if hull.shape[0] < 3:
        return 0.0

    x = hull[:, 0]
    y = hull[:, 1]

    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(area)


def select_nearest_gate_per_beam(
    Ne_2d,
    x_km,
    y_km,
    z_km,
    altitude_km,
    h0_km,
    half_width_km,
    beamcodes,
    dNe_2d=None,
):
    """
    For each beam, select the valid range gate closest to h0_km.

    This gives at most one point per beam.
    """

    rows = []

    nbeam, nrange = Ne_2d.shape

    for ibeam in range(nbeam):
        alt = altitude_km[ibeam, :]
        ne = Ne_2d[ibeam, :]

        valid = np.isfinite(ne)
        valid &= ne > 0
        valid &= np.isfinite(alt)
        valid &= np.isfinite(x_km[ibeam, :])
        valid &= np.isfinite(y_km[ibeam, :])
        valid &= np.isfinite(z_km[ibeam, :])
        valid &= np.abs(alt - h0_km) <= half_width_km

        if dNe_2d is not None:
            dne = dNe_2d[ibeam, :]
            valid &= np.isfinite(dne)
            valid &= dne > 0

        idx = np.where(valid)[0]

        if idx.size == 0:
            continue

        local_best = np.argmin(np.abs(alt[idx] - h0_km))
        irange = int(idx[local_best])

        row = {
            "beam_index": ibeam,
            "beamcode": beamcodes[ibeam],
            "range_index": irange,
            "x_km": x_km[ibeam, irange],
            "y_km": y_km[ibeam, irange],
            "z_km": z_km[ibeam, irange],
            "altitude_km": altitude_km[ibeam, irange],
            "dz_from_h0_km": altitude_km[ibeam, irange] - h0_km,
            "Ne": Ne_2d[ibeam, irange],
            "log10_Ne": np.log10(Ne_2d[ibeam, irange]),
        }

        if dNe_2d is not None:
            row["dNe"] = dNe_2d[ibeam, irange]
            row["rel_dNe"] = dNe_2d[ibeam, irange] / Ne_2d[ibeam, irange]

        rows.append(row)

    return pd.DataFrame(rows)


def scan_altitude_bands(
    Ne_2d,
    x_km,
    y_km,
    z_km,
    altitude_km,
    beamcodes,
    h_min_km,
    h_max_km,
    h_step_km,
    half_width_km,
    min_beams,
    dNe_2d=None,
):
    h_values = np.arange(h_min_km, h_max_km + 0.5 * h_step_km, h_step_km)

    rows = []
    slices = {}

    total_beams = Ne_2d.shape[0]

    for h0 in h_values:
        df = select_nearest_gate_per_beam(
            Ne_2d=Ne_2d,
            x_km=x_km,
            y_km=y_km,
            z_km=z_km,
            altitude_km=altitude_km,
            h0_km=h0,
            half_width_km=half_width_km,
            beamcodes=beamcodes,
            dNe_2d=dNe_2d,
        )

        n_points = len(df)
        n_beams = df["beamcode"].nunique() if n_points > 0 else 0

        if n_points > 0:
            area_km2 = convex_hull_area_km2(df[["x_km", "y_km"]].to_numpy())
            dz_rms_km = float(np.sqrt(np.mean(df["dz_from_h0_km"].to_numpy() ** 2)))
            dz_max_abs_km = float(np.max(np.abs(df["dz_from_h0_km"].to_numpy())))
            log10_ne_std = float(df["log10_Ne"].std(ddof=0))
        else:
            area_km2 = 0.0
            dz_rms_km = np.nan
            dz_max_abs_km = np.nan
            log10_ne_std = np.nan

        if "rel_dNe" in df.columns and n_points > 0:
            median_rel_dNe = float(np.nanmedian(df["rel_dNe"].to_numpy()))
        else:
            median_rel_dNe = np.nan

        rows.append(
            {
                "h0_km": float(h0),
                "half_width_km": float(half_width_km),
                "n_points": int(n_points),
                "n_beams": int(n_beams),
                "beam_fraction": float(n_beams / total_beams),
                "area_km2": float(area_km2),
                "dz_rms_km": dz_rms_km,
                "dz_max_abs_km": dz_max_abs_km,
                "median_rel_dNe": median_rel_dNe,
                "log10_Ne_std": log10_ne_std,
            }
        )

        slices[float(h0)] = df

    metrics = pd.DataFrame(rows)

    max_area = metrics["area_km2"].max()
    if not np.isfinite(max_area) or max_area <= 0:
        max_area = 1.0

    metrics["area_norm"] = metrics["area_km2"] / max_area

    if np.isfinite(metrics["median_rel_dNe"]).any():
        metrics["uncertainty_penalty"] = metrics["median_rel_dNe"].clip(lower=0.0, upper=1.0)
        metrics["uncertainty_penalty"] = metrics["uncertainty_penalty"].fillna(1.0)
    else:
        metrics["uncertainty_penalty"] = 0.0

    metrics["dz_penalty"] = (metrics["dz_rms_km"] / half_width_km).clip(lower=0.0, upper=1.0)
    metrics["dz_penalty"] = metrics["dz_penalty"].fillna(1.0)

    metrics["score"] = (
        5.0 * metrics["beam_fraction"]
        + 1.0 * metrics["area_norm"]
        - 1.0 * metrics["uncertainty_penalty"]
        - 0.5 * metrics["dz_penalty"]
    )

    metrics.loc[metrics["n_beams"] < min_beams, "score"] = -np.inf

    if np.isneginf(metrics["score"]).all():
        raise RuntimeError(
            f"No altitude band met min_beams={min_beams}. "
            f"Try increasing --half_width_km."
        )

    metrics_sorted = metrics.sort_values("score", ascending=False).reset_index(drop=True)

    best_h0 = float(metrics_sorted.iloc[0]["h0_km"])
    best_slice = slices[best_h0].copy()

    return metrics_sorted, best_slice


def normalize_to_minus1_plus1(values):
    values = np.asarray(values, dtype=float)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    if vmax <= vmin:
        raise ValueError(f"Cannot normalize because vmax <= vmin: {vmax} <= {vmin}")

    norm = 2.0 * (values - vmin) / (vmax - vmin) - 1.0

    return norm, vmin, vmax


def add_inr_columns(best_slice):
    df = best_slice.copy()

    df["x_norm"], x_min, x_max = normalize_to_minus1_plus1(df["x_km"].to_numpy())
    df["y_norm"], y_min, y_max = normalize_to_minus1_plus1(df["y_km"].to_numpy())
    df["log10_Ne_norm"], ne_min, ne_max = normalize_to_minus1_plus1(df["log10_Ne"].to_numpy())

    coords = df[["x_norm", "y_norm"]].to_numpy(dtype=np.float32)
    values = df[["log10_Ne_norm"]].to_numpy(dtype=np.float32)

    scaler = {
        "x_min_km": x_min,
        "x_max_km": x_max,
        "y_min_km": y_min,
        "y_max_km": y_max,
        "log10_Ne_min": ne_min,
        "log10_Ne_max": ne_max,
    }

    return df, coords, values, scaler


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("filepath", type=str)

    parser.add_argument("--time_index", type=int, default=0)
    parser.add_argument("--h_min_km", type=float, default=150.0)
    parser.add_argument("--h_max_km", type=float, default=500.0)
    parser.add_argument("--h_step_km", type=float, default=5.0)
    parser.add_argument("--half_width_km", type=float, default=5.0)
    parser.add_argument("--min_beams", type=int, default=35)

    parser.add_argument("--range_units", type=str, default="auto", choices=["auto", "m", "km"])
    parser.add_argument("--altitude_units", type=str, default="auto", choices=["auto", "m", "km"])

    parser.add_argument("--out_prefix", type=str, default="amisr_2d_slice")

    args = parser.parse_args()

    data = read_amisr_h5(args.filepath)
    print_summary(data)

    Ne = data["Ne"]

    if Ne.ndim != 3:
        raise ValueError(f"Expected Ne shape [time, beam, range], got {Ne.shape}")

    nt, nb, nr = Ne.shape

    if args.time_index < 0 or args.time_index >= nt:
        raise ValueError(f"time_index={args.time_index} outside valid range 0 to {nt - 1}")

    print()
    print(f"Using time_index = {args.time_index}")
    print(f"Unix midpoint = {data['unix_mid'][args.time_index]}")
    print(
        "UTC midpoint =",
        datetime.fromtimestamp(data["unix_mid"][args.time_index], tz=timezone.utc),
    )

    x_km, y_km, z_km, range_km, altitude_km = make_xyz_for_time(
        data,
        time_index=args.time_index,
        range_units=args.range_units,
        altitude_units=args.altitude_units,
    )

    Ne_2d = data["Ne"][args.time_index, :, :]

    if data["dNe"] is not None:
        dNe_2d = data["dNe"][args.time_index, :, :]
    else:
        dNe_2d = None

    metrics, best_slice = scan_altitude_bands(
        Ne_2d=Ne_2d,
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        altitude_km=altitude_km,
        beamcodes=data["beamcodes"],
        h_min_km=args.h_min_km,
        h_max_km=args.h_max_km,
        h_step_km=args.h_step_km,
        half_width_km=args.half_width_km,
        min_beams=args.min_beams,
        dNe_2d=dNe_2d,
    )

    best_df, coords, values, scaler = add_inr_columns(best_slice)

    out_prefix = Path(args.out_prefix)

    metrics_file = out_prefix.with_name(out_prefix.name + "_altitude_scan.csv")
    slice_file = out_prefix.with_name(out_prefix.name + "_best_slice.csv")
    npz_file = out_prefix.with_name(out_prefix.name + "_inr_arrays.npz")

    metrics.to_csv(metrics_file, index=False)
    best_df.to_csv(slice_file, index=False)

    np.savez(
        npz_file,
        coords=coords,
        values=values,
        x_km=best_df["x_km"].to_numpy(dtype=np.float32),
        y_km=best_df["y_km"].to_numpy(dtype=np.float32),
        z_km=best_df["z_km"].to_numpy(dtype=np.float32),
        altitude_km=best_df["altitude_km"].to_numpy(dtype=np.float32),
        log10_Ne=best_df["log10_Ne"].to_numpy(dtype=np.float32),
        beamcode=best_df["beamcode"].to_numpy(),
        range_index=best_df["range_index"].to_numpy(),
        scaler=np.array([scaler], dtype=object),
    )

    print()
    print("Best altitude candidate:")
    print(metrics.head(10))

    print()
    print("Selected slice:")
    print(f"  rows:         {len(best_df)}")
    print(f"  unique beams: {best_df['beamcode'].nunique()}")
    print(f"  coords shape: {coords.shape}")
    print(f"  values shape: {values.shape}")

    print()
    print("Saved:")
    print(f"  {metrics_file}")
    print(f"  {slice_file}")
    print(f"  {npz_file}")


if __name__ == "__main__":
    main()