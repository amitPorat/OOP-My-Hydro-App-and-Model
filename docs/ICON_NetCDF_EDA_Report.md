# ICON Raw NetCDF — Exploratory Data Analysis Report

**File:** `/media/data-nvme/icon/ICON_ENS_2026011200.nc`  
**Tool:** `scripts/inspect_icon_nc.py` (xarray, read-only)  
**Purpose:** Inform design of the ICON processing pipeline in `src/rain_builder.py`.

---

## 1. Dimensions

| Dimension | Size | Notes |
|-----------|------|--------|
| **time**  | 181  | Main time axis |
| **lat**   | 197  | Latitude points |
| **lon**   | 115  | Longitude points |
| **depth** | 8    | Soil layers (e.g. w_so) |
| **depth_2** | 9  | Alternative depth (e.g. t_so) |
| **height** | 1   | 2 m level (e.g. T2, Q2) |
| **height_2** | 1 | 10 m level (e.g. U10, V10) |
| **bnds**  | 2    | Bounds for depth_bnds |

Primary grid for precipitation and 2D fields: **(time, lat, lon)** → 181 × 197 × 115.

---

## 2. Variables

### Precipitation (rain)

- **Names:** `RAINC_01`, `RAINC_02`, …, `RAINC_20` (20 ensemble members).
- **Dimensions:** `(time, lat, lon)` — shape **(181, 197, 115)**.
- **Dtype:** float32.
- **Units:** `kg m-2` (equivalent to mm depth).
- **long_name:** `total precip`.

No separate convective/large-scale split in this file; single “total precip” per member.

### Other variables (per member)

- **w_so_XX:** Total water content (ice + liquid), `(time, depth, lat, lon)`.
- **Q2_XX, T2_XX, U10_XX, V10_XX, PSFC_XX, GLW_XX, SWDOWN_XX, t_g_XX, t_so_XX:** Various surface/2 m/10 m fields (see script output for units).

---

## 3. Time Resolution and Type

- **Coordinate:** `time`.
- **Length:** 181 steps.
- **First:** 2026-01-12 00:00:00  
- **Last:** 2026-01-15 18:00:00  
- **Step:** **30 minutes** (0 days 00:30:00).

**Span:** ~3.75 days at 30-min resolution.

**Precipitation type:**  
- Stored as **total precip** in `kg m-2` (mm).  
- From the file history, the original ICON variable was **tot_prec** (total precipitation). In ICON/NWP, this is usually **accumulated since model start** (or since last reset). So each time step typically holds a **running cumulative** value; to get **per-period accumulation** or **rate** you must **difference** along `time` (e.g. `RAINC(t) - RAINC(t-1)`), then optionally convert to mm/h using the 30-min step (e.g. multiply by 2 for rate in mm/h).

---

## 4. Spatial Resolution and Bounding Box

| Coordinate | Min    | Max    | Size | Approx. spacing |
|------------|--------|--------|------|------------------|
| **lat**    | 29.0°  | 33.9°  | 197  | **0.025°** (~2.8 km) |
| **lon**    | 33.85° | 36.7°  | 115  | **0.025°** (~2.8 km) |

- **Bounding box (WGS84):** lon [33.85, 36.7], lat [29, 33.9] (covers Israel and vicinity).
- **Grid:** Regular lat/lon, ~0.025° spacing (CF-style 1D lat/lon).

---

## 5. CRS / Projection

- **No explicit CRS or grid_mapping** in coordinates or variables.
- **Interpretation:** Latitude and longitude are in **degrees**; standard practice for this format is **WGS84 (EPSG:4326)**. Safe to assume EPSG:4326 for reprojection or comparison with RMCOMP/radar.

---

## 6. Pipeline Design Notes

1. **Precipitation:** Use `RAINC_01` … `RAINC_20`; decide whether to use one member, mean, or full ensemble. Convert cumulative to per-step accumulation (diff in time), then to rate (mm/h) if needed to match RMCOMP/radar rate.
2. **Time:** 30-min steps; align with your 10-min RMCOMP series (e.g. resample or interpolate as needed).
3. **Space:** Regular 0.025° lat/lon; regridding to your UGRID/catchment grid will require a separate step (e.g. nearest-neighbour or conservative).
4. **Units:** `kg m-2` = mm; ensure consistency with existing rain rate units (e.g. mm/h) when differencing.

---

*Report generated from `scripts/inspect_icon_nc.py`; no modifications were made to the NetCDF file.*
