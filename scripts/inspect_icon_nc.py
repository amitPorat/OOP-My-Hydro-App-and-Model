#!/usr/bin/env python3
"""
Temporary script: inspect raw ICON NetCDF file structure for EDA.
Read-only; no modifications. Output is formatted for pipeline design.
"""
from pathlib import Path

import xarray as xr

ICON_PATH = Path("/media/data-nvme/icon/ICON_ENS_2026011200.nc")


def main() -> None:
    if not ICON_PATH.exists():
        print(f"File not found: {ICON_PATH}")
        return
    ds = xr.open_dataset(ICON_PATH)
    print("=" * 70)
    print("ICON RAW NetCDF — EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 70)
    print(f"\nFile: {ICON_PATH}")
    print(f"File size: {ICON_PATH.stat().st_size / (1024**2):.2f} MB")

    # 1. Dimensions
    print("\n" + "-" * 70)
    print("1. DIMENSIONS (coordinate names and sizes)")
    print("-" * 70)
    for name, size in ds.sizes.items():
        print(f"   {name}: {size}")

    # 2. Variables (all, with precipitation highlighted)
    print("\n" + "-" * 70)
    print("2. VARIABLES (names, dtype, dimensions, shape)")
    print("-" * 70)
    precip_keywords = ["rain", "precip", "RAIN", "PRECIP", "pr", "tp", "apcp"]
    for name, var in ds.data_vars.items():
        is_precip = any(kw in name for kw in precip_keywords)
        marker = "  [PRECIP?]" if is_precip else ""
        dims = var.dims
        shape = var.shape
        dtype = str(var.dtype)
        print(f"   {name}{marker}")
        print(f"      dims: {dims}  shape: {shape}  dtype: {dtype}")
    print("\n   Coordinates (non-dimension):")
    for name, var in ds.coords.items():
        if name not in ds.dims:
            print(f"   {name}: dims={var.dims}, shape={var.shape}, dtype={var.dtype}")

    # 3. Time resolution and type
    print("\n" + "-" * 70)
    print("3. TIME RESOLUTION & TYPE")
    print("-" * 70)
    time_coords = [c for c in ds.coords if "time" in c.lower()]
    for tc in time_coords:
        coord = ds.coords[tc]
        print(f"   Coordinate: {tc}")
        try:
            vals = coord.values
            print(f"     length: {len(vals)}")
            if len(vals) > 0:
                print(f"     first: {vals[0]}")
                print(f"     last:  {vals[-1]}")
            if len(vals) >= 2:
                diff = coord.to_index().to_series().diff().dropna()
                if len(diff) > 0:
                    print(f"     sample step: {diff.iloc[0]}")
        except Exception as e:
            print(f"     (could not read values: {e})")
    # Check for time in variable dims and units
    for name, var in ds.data_vars.items():
        if "time" in var.dims and hasattr(var, "attrs"):
            a = var.attrs
            if "units" in a or "long_name" in a:
                print(f"   Variable '{name}' attrs: units={a.get('units')}, long_name={a.get('long_name')}")

    # 4. Spatial resolution and bounding box
    print("\n" + "-" * 70)
    print("4. SPATIAL RESOLUTION & BOUNDING BOX")
    print("-" * 70)
    lat_cands = [c for c in ds.coords if "lat" in c.lower() or "y" == c.lower() or "latitude" in c.lower()]
    lon_cands = [c for c in ds.coords if "lon" in c.lower() or "x" == c.lower() or "longitude" in c.lower()]
    for name in lat_cands:
        c = ds.coords[name]
        try:
            v = c.values
            if v.size > 0:
                print(f"   {name}: min={float(v.min()):.4f}, max={float(v.max()):.4f}, size={v.size}")
                if v.size >= 2:
                    spacing = float(abs(v.flat[1] - v.flat[0]))
                    print(f"          approximate spacing: {spacing:.4f} deg")
        except Exception as e:
            print(f"   {name}: (read error: {e})")
    for name in lon_cands:
        c = ds.coords[name]
        try:
            v = c.values
            if v.size > 0:
                print(f"   {name}: min={float(v.min()):.4f}, max={float(v.max()):.4f}, size={v.size}")
                if v.size >= 2:
                    spacing = float(abs(v.flat[1] - v.flat[0]))
                    print(f"          approximate spacing: {spacing:.4f} deg")
        except Exception as e:
            print(f"   {name}: (read error: {e})")
    # Also check for 2D lat/lon
    for name, var in ds.data_vars.items():
        if "lat" in name.lower() or "lon" in name.lower():
            if hasattr(var.values, "min"):
                v = var.values
                print(f"   Data var {name}: shape={v.shape}, min={float(v.min()):.4f}, max={float(v.max()):.4f}")

    # 5. CRS / projection
    print("\n" + "-" * 70)
    print("5. CRS / PROJECTION")
    print("-" * 70)
    crs_found = False
    for name in list(ds.coords) + list(ds.data_vars):
        obj = ds.coords.get(name, None) if name in ds.coords else ds.data_vars.get(name, None)
        if obj is not None and hasattr(obj, "attrs"):
            for k, v in obj.attrs.items():
                if any(x in k.lower() for x in ["crs", "projection", "grid_mapping", "epsg", "wgs"]):
                    print(f"   {name}.{k}: {v}")
                    crs_found = True
    if not crs_found:
        for k, v in ds.attrs.items():
            if any(x in k.lower() for x in ["crs", "projection", "grid_mapping", "epsg", "wgs"]):
                print(f"   global attr {k}: {v}")
                crs_found = True
    if not crs_found:
        print("   No explicit CRS/projection attributes found (often implies WGS84 / EPSG:4326 for lat/lon in deg).")

    # Global attributes summary
    print("\n" + "-" * 70)
    print("6. GLOBAL ATTRIBUTES (summary)")
    print("-" * 70)
    for k, v in list(ds.attrs.items())[:30]:
        print(f"   {k}: {v}")
    if len(ds.attrs) > 30:
        print(f"   ... and {len(ds.attrs) - 30} more")

    ds.close()
    print("\n" + "=" * 70)
    print("End of EDA report")
    print("=" * 70)


if __name__ == "__main__":
    main()
