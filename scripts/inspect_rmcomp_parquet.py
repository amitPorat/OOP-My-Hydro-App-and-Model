#!/usr/bin/env python3
"""
Profile a sample RMCOMP (kriged rain on UGRID) parquet for EDA.
Read-only; used to compare with ICON and define transformation requirements.
"""
from pathlib import Path

import pandas as pd

# Prefer kriged rain (post-processed on UGRID) as the target RMCOMP structure
SAMPLE_PATHS = [
    Path("/media/data-nvme/Darga_28_for_test_only/output/rain/event_rain/intepulated_rain_on_ugrid/rain_kriged_2016.parquet"),
    Path("/media/data-nvme/Darga_28_for_test_only/output/rain/event_rain/intepulated_rain_on_ugrid/rain_kriged_2015.parquet"),
    Path("/media/data-nvme/Darga_28_for_test_only/output/rain/rain_2016.parquet"),
    Path("/media/data-nvme/Darga_28_for_test_only/output/rain_with_discharge/rain_with_discharge_2016.parquet"),
]


def main() -> None:
    path = None
    for p in SAMPLE_PATHS:
        if p.exists():
            path = p
            break
    if path is None:
        print("No sample RMCOMP parquet found at any of:", [str(p) for p in SAMPLE_PATHS])
        return

    print("=" * 70)
    print("RMCOMP (target) Parquet — Structure & Time Profile")
    print("=" * 70)
    print(f"\nFile: {path}")
    print(f"Size: {path.stat().st_size / (1024**2):.2f} MB")

    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])

    # 1. Column names and dtypes
    print("\n" + "-" * 70)
    print("1. COLUMN NAMES & DTYPES")
    print("-" * 70)
    for c in df.columns:
        print(f"   {c}: {df[c].dtype}  (n={df[c].count()}, nulls={df[c].isna().sum()})")

    # 2. Time frequency per ID
    print("\n" + "-" * 70)
    print("2. TIME FREQUENCY (per ID)")
    print("-" * 70)
    id_col = "ID" if "ID" in df.columns else None
    if id_col:
        ids = df[id_col].dropna().unique()
        sample_ids = ids[:5] if len(ids) >= 5 else ids
        all_deltas = []
        for cell_id in sample_ids:
            sub = df[df[id_col] == cell_id].sort_values("time")
            t = sub["time"]
            if len(t) >= 2:
                deltas = t.diff().dropna()
                all_deltas.extend(deltas.tolist())
                mode_delta = deltas.mode()
                print(f"   ID={cell_id}: rows={len(sub)}, time range [{t.min()} .. {t.max()}], "
                      f"diff sample: {deltas.iloc[0]}, min={deltas.min()}, max={deltas.max()}")
        if all_deltas:
            s = pd.Series(all_deltas)
            print(f"   Overall (sample IDs): median diff = {s.median()}, mode = {s.mode().iloc[0] if len(s.mode()) else 'N/A'}")
    else:
        df_sorted = df.sort_values("time")
        t = df_sorted["time"]
        if len(t) >= 2:
            deltas = t.diff().dropna()
            print(f"   Single series: rows={len(df)}, diff sample = {deltas.iloc[0]}, median = {deltas.median()}")

    # 3. Rain variable: units / value stats
    print("\n" + "-" * 70)
    print("3. RAIN VARIABLE (units / value stats)")
    print("-" * 70)
    rain_col = "rainrate" if "rainrate" in df.columns else None
    if rain_col is None:
        for c in df.columns:
            if "rain" in c.lower():
                rain_col = c
                break
    if rain_col:
        r = df[rain_col]
        print(f"   Column: '{rain_col}'")
        print(f"   min={r.min():.6f}, max={r.max():.6f}, mean={r.mean():.6f}, median={r.median():.6f}")
        print(f"   >0 count: {(r > 0).sum()}, zeros: {(r == 0).sum()}")
        # If we have time + ID, rough rate check: mm/interval vs mm/h
        if id_col and len(df) >= 2:
            sub = df[df[id_col] == df[id_col].iloc[0]].sort_values("time").head(20)
            if len(sub) >= 2:
                dt_min = (sub["time"].diff().dropna().min().total_seconds() / 60.0)  # minutes
                print(f"   (Time step for one ID sample: ~{dt_min:.0f} min)")
    else:
        print("   No obvious rain column found.")

    # 4. Spatial: ID count, optional lon/lat
    print("\n" + "-" * 70)
    print("4. SPATIAL FORMAT")
    print("-" * 70)
    if id_col:
        n_ids = df[id_col].nunique()
        print(f"   Unique {id_col}: {n_ids}")
    if "lon" in df.columns and "lat" in df.columns:
        print(f"   lon range: [{df['lon'].min():.4f}, {df['lon'].max():.4f}]")
        print(f"   lat range: [{df['lat'].min():.4f}, {df['lat'].max():.4f}]")
    else:
        print("   (No lon/lat in this file; merged product adds them from UGRID.)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
