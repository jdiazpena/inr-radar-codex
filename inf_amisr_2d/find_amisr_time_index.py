# -*- coding: utf-8 -*-

import argparse
from datetime import datetime, timezone, time
from pathlib import Path

import h5py
import numpy as np


def read_unix_time(filepath):
    filepath = Path(filepath)

    with h5py.File(filepath, "r") as f:
        if "Time/UnixTime" not in f:
            raise KeyError("Could not find Time/UnixTime in this HDF5 file.")

        unix_time = np.asarray(f["Time/UnixTime"], dtype=float)

    if unix_time.ndim != 2 or unix_time.shape[1] < 2:
        raise ValueError(f"Expected Time/UnixTime shape [ntime, 2], got {unix_time.shape}")

    unix_start = unix_time[:, 0]
    unix_end = unix_time[:, 1]
    unix_mid = 0.5 * (unix_start + unix_end)

    return unix_start, unix_mid, unix_end


def parse_target_time(unix_mid, target_utc=None, target_hhmmss=None):
    if target_utc is None and target_hhmmss is None:
        raise ValueError("Use either --target_utc or --target_hhmmss.")

    if target_utc is not None:
        s = target_utc.replace("T", " ")
        dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp(), dt

    # Infer date from first radar record.
    first_dt = datetime.fromtimestamp(float(unix_mid[0]), tz=timezone.utc)
    date0 = first_dt.date()

    hh, mm, ss = target_hhmmss.split(":")
    dt = datetime.combine(
        date0,
        time(int(hh), int(mm), int(ss)),
        tzinfo=timezone.utc,
    )

    return dt.timestamp(), dt


def find_time_index(filepath, target_utc=None, target_hhmmss=None):
    unix_start, unix_mid, unix_end = read_unix_time(filepath)

    target_unix, target_dt = parse_target_time(
        unix_mid,
        target_utc=target_utc,
        target_hhmmss=target_hhmmss,
    )

    inside = np.where((unix_start <= target_unix) & (target_unix <= unix_end))[0]

    if inside.size > 0:
        best_local = np.argmin(np.abs(unix_mid[inside] - target_unix))
        idx = int(inside[best_local])
        mode = "target is inside integration interval"
    else:
        idx = int(np.argmin(np.abs(unix_mid - target_unix)))
        mode = "nearest midpoint"

    start_dt = datetime.fromtimestamp(float(unix_start[idx]), tz=timezone.utc)
    mid_dt = datetime.fromtimestamp(float(unix_mid[idx]), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(float(unix_end[idx]), tz=timezone.utc)

    print("Target:")
    print(f"  UTC:   {target_dt}")
    print(f"  Unix:  {target_unix:.3f}")
    print()
    print("Selected AMISR record:")
    print(f"  time_index: {idx}")
    print(f"  mode:       {mode}")
    print(f"  start UTC:  {start_dt}")
    print(f"  mid UTC:    {mid_dt}")
    print(f"  end UTC:    {end_dt}")
    print(f"  midpoint difference from target: {unix_mid[idx] - target_unix:.3f} s")
    print()
    print("Command value to use:")
    print(f"  --time_index {idx}")

    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filepath", type=str, help="AMISR HDF5 file")
    parser.add_argument("--target_utc", type=str, default=None, help='Example: "YYYY-MM-DD 11:17:48"')
    parser.add_argument("--target_hhmmss", type=str, default=None, help='Example: "11:17:48"')

    args = parser.parse_args()

    find_time_index(
        args.filepath,
        target_utc=args.target_utc,
        target_hhmmss=args.target_hhmmss,
    )


if __name__ == "__main__":
    main()