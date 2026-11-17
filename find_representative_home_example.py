#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_ROOT = Path(
    "/shared/Digital Clinic/cortex2_outputs/digital_clinic"
)


def find_single(root: Path, filename: str) -> Path:
    hits = sorted(
        root.rglob(filename),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(
            f"Could not find {filename!r} under {root}"
        )
    print(f"[INFO] Using: {hits[0]}")
    return hits[0]


def pick_col(df: pd.DataFrame, candidates, label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not find {label}. Tried: {candidates}\n"
        f"Available columns: {list(df.columns)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--start",
        default="2025-04-23",
        help="Window start date (inclusive)",
    )
    parser.add_argument(
        "--end",
        default="2025-05-07",
        help="Window end date (exclusive)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--min-supported-days",
        type=int,
        default=8,
    )
    args = parser.parse_args()

    daily_path = find_single(args.output_root, "daily_home_time.csv")
    df = pd.read_csv(daily_path)

    pid_col = pick_col(
        df,
        ["participant_id", "participant", "lamp_id", "id"],
        "participant column",
    )
    date_col = pick_col(
        df,
        ["date", "local_date", "day", "study_date"],
        "date column",
    )
    home_col = pick_col(
        df,
        ["home_duration_ms"],
        "home duration column",
    )
    supported_col = pick_col(
        df,
        ["supported_duration_ms", "observed_duration_ms"],
        "supported duration column",
    )

    frac_col = None
    for c in ["time_at_home_fraction", "home_fraction"]:
        if c in df.columns:
            frac_col = c
            break

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    df = df[
        (df[date_col] >= start) &
        (df[date_col] < end)
    ].copy()

    if df.empty:
        raise RuntimeError("No rows found in selected date window.")

    df[home_col] = pd.to_numeric(df[home_col], errors="coerce")
    df[supported_col] = pd.to_numeric(df[supported_col], errors="coerce")

    n_days_window = (end - start).days

    agg = (
        df.groupby(pid_col)
        .agg(
            n_days_rows=(date_col, "size"),
            n_days_supported=(supported_col, lambda s: int((s.fillna(0) > 0).sum())),
            total_home_ms=(home_col, "sum"),
            total_supported_ms=(supported_col, "sum"),
        )
        .reset_index()
    )

    agg["window_days"] = n_days_window
    agg["total_home_h"] = agg["total_home_ms"] / 3_600_000.0
    agg["total_supported_h"] = agg["total_supported_ms"] / 3_600_000.0
    agg["total_unobserved_h"] = (24.0 * n_days_window) - agg["total_supported_h"]
    agg["home_fraction_supported"] = np.where(
        agg["total_supported_h"] > 0,
        agg["total_home_h"] / agg["total_supported_h"],
        np.nan,
    )
    agg["mean_supported_h_per_supported_day"] = np.where(
        agg["n_days_supported"] > 0,
        agg["total_supported_h"] / agg["n_days_supported"],
        np.nan,
    )

    usable = agg[
        (agg["n_days_supported"] >= args.min_supported_days) &
        (agg["total_home_h"] > 0) &
        (agg["total_supported_h"] > 0)
    ].copy()

    if usable.empty:
        raise RuntimeError(
            "No usable participants after filters. "
            "Try lowering --min-supported-days."
        )

    med_days = usable["n_days_supported"].median()
    med_supported = usable["total_supported_h"].median()
    med_home_frac = usable["home_fraction_supported"].median()

    usable["score"] = (
        (usable["n_days_supported"] - med_days).abs() / max(med_days, 1)
        + (usable["total_supported_h"] - med_supported).abs() / max(med_supported, 1)
        + (usable["home_fraction_supported"] - med_home_frac).abs() / max(med_home_frac, 1e-6)
    )

    usable = usable.sort_values(
        ["score", "n_days_supported", "total_supported_h"],
        ascending=[True, False, False],
    )

    print("\n" + "=" * 78)
    print("REPRESENTATIVE HOME-TIME EXAMPLE SEARCH")
    print("=" * 78)
    print(f"Window: {start.date()} to {end.date()} ({n_days_window} days)")
    print(f"Participants in window: {agg.shape[0]}")
    print(f"Usable participants: {usable.shape[0]}")
    print()
    print("Cohort medians among usable participants")
    print(f"  Supported days: {med_days:.1f}")
    print(f"  Total supported hours: {med_supported:.1f}")
    print(f"  Home fraction of supported time: {med_home_frac:.3f}")
    print()

    cols = [
        pid_col,
        "n_days_supported",
        "total_supported_h",
        "total_home_h",
        "total_unobserved_h",
        "home_fraction_supported",
        "mean_supported_h_per_supported_day",
        "score",
    ]

    show = usable[cols].head(args.top).copy()
    for c in [
        "total_supported_h",
        "total_home_h",
        "total_unobserved_h",
        "home_fraction_supported",
        "mean_supported_h_per_supported_day",
        "score",
    ]:
        show[c] = show[c].round(3)

    print(f"Top {min(args.top, len(show))} candidate participants")
    print(show.to_string(index=False))

    if frac_col and frac_col in df.columns:
        print()
        print(f"[INFO] Daily fraction column present: {frac_col}")

    best = usable.iloc[0]
    print("\n" + "-" * 78)
    print("BEST CANDIDATE")
    print("-" * 78)
    print(f"Participant: {best[pid_col]}")
    print(f"Supported days: {int(best['n_days_supported'])}/{n_days_window}")
    print(f"Supported hours: {best['total_supported_h']:.1f}")
    print(f"Home hours: {best['total_home_h']:.1f}")
    print(f"Unobserved hours: {best['total_unobserved_h']:.1f}")
    print(f"Home fraction of supported time: {best['home_fraction_supported']:.3f}")


if __name__ == "__main__":
    main()
