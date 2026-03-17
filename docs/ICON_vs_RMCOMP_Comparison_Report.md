# ICON vs RMCOMP — Comparison Report

**Purpose:** Define the target RMCOMP structure and the exact transformations required in `src/rain_builder.py` so that ICON-derived rain matches RMCOMP format and semantics. No assumptions about their equivalence.

**Sources:**
- **ICON:** `/media/data-nvme/icon/ICON_ENS_2026011200.nc` (see `docs/ICON_NetCDF_EDA_Report.md`).
- **RMCOMP (target):** Sample `rain_kriged_2016.parquet` under  
  `/media/data-nvme/Darga_28_for_test_only/output/rain/event_rain/intepulated_rain_on_ugrid/`  
  (inspected via `scripts/inspect_rmcomp_parquet.py`).

---

## 1. Target RMCOMP Structure (Profiled)

| Aspect | RMCOMP (rain_kriged_*.parquet) |
|--------|--------------------------------|
| **Columns** | `ID` (int32), `time` (datetime64[ns]), `rainrate` (float32). No lon/lat in this file; the merger adds them from UGRID when producing `rain_with_discharge_*.parquet`. |
| **Time frequency** | **10 minutes.** Median and mode of inter-step difference = 10 min. (Some larger gaps exist due to event-based slicing.) |
| **Rain variable** | `rainrate`: **depth per 10‑minute interval** in **mm** (i.e. mm/10min). Notebook label: “Rain Rate (mm/10min)”. Values: float32, non‑negative; observed max ~5.85, many zeros. |
| **Spatial format** | **1D list of UGRID cell IDs.** One row per (ID, time); 60,739 unique IDs in sample. Grid is the catchment UGRID; lon/lat come from a separate UGRID parquet when needed. |
| **File format** | Parquet (optional columns for merger: ID, time, rainrate; merger adds discharge, lon, lat from UGRID). |

So the **exact target** for ICON-derived rain is:

- **Schema:** `ID` (int32), `time` (datetime64), `rainrate` (float32).
- **Semantics:** `rainrate` = **mm of rain in that 10‑minute interval** (depth per 10 min, not mm/h).
- **Time axis:** Regular(ish) **10‑min** steps per ID.
- **Space:** One row per (UGRID cell ID, time).

---

## 2. Comparison Matrix: ICON vs RMCOMP

| Aspect | ICON (raw) | RMCOMP (target) |
|--------|------------|------------------|
| **Time resolution** | **30 minutes** (181 steps, 30 min apart). | **10 minutes** (10 min between rows per ID). |
| **Data type (precip)** | **Cumulative** since model start: `RAINC_XX` in **kg m⁻²** (= mm), running total. Per-step accumulation = diff in time. | **Discrete depth per interval:** `rainrate` in **mm per 10‑min** (depth in that 10‑min window). |
| **Spatial format** | **2D regular grid:** 1D coordinates `lat` (197), `lon` (115); spacing ~0.025°; array shape `(time, lat, lon)`. | **1D list of UGRID cell IDs:** one row per (ID, time); IDs reference a separate UGRID (e.g. `final_ugrid.parquet`) with lon/lat. |
| **File format** | NetCDF (xarray). | Parquet (pandas/pyarrow): ID, time, rainrate. |
| **Ensemble** | 20 members: `RAINC_01` … `RAINC_20`. | Single series (no ensemble in this parquet). |
| **CRS** | Implicit WGS84 (lat/lon in deg). | Same (UGRID cells in WGS84). |

---

## 3. Transformations Required in `src/rain_builder.py`

To make ICON output **match RMCOMP exactly** (schema, time, units, space), apply these steps in order.

### 3.1 Precipitation: Cumulative → Depth per 10 min (same units as RMCOMP)

1. **Cumulative → per-step depth (30 min)**  
   - For chosen member(s), e.g. `RAINC_01`:  
     `accum_30min(t) = RAINC(t) - RAINC(t-1)` (with `RAINC(t0) = 0` or first step as-is).  
   - Units: **mm per 30 min** (kg m⁻² = mm).

2. **30‑min depth → 10‑min depth (align with RMCOMP “rainrate”)**
   - RMCOMP stores **mm per 10‑min interval**.  
   - Options (to be chosen in implementation):
     - **A. Uniform split:** `rainrate_10min = accum_30min / 3` at each 30‑min step, then assign to the three 10‑min timestamps (t, t+10, t+20).  
     - **B. Resample / interpolate:** Build a 10‑min time index, interpolate 30‑min accumulations to 10‑min (e.g. backward sum or linear), so that three 10‑min values sum to the 30‑min accumulation.  
   - Output variable must be **float32**, **non‑negative**, in **mm per 10‑min** so it is directly comparable to RMCOMP `rainrate`.

### 3.2 Time: 30‑min axis → 10‑min axis

3. **Generate 10‑min time series**  
   - Target: 10‑min steps from forecast start to end.  
   - Either expand each 30‑min step to three 10‑min steps (with the depth conversion above), or build a full 10‑min `datetime` index and fill from the 30‑min series.  
   - Align timezone/epoch with RMCOMP (typically UTC or local; ensure no double DST shift).

### 3.3 Space: 2D lat/lon → 1D UGRID IDs

4. **Regrid from ICON (lat, lon) to UGRID cells**  
   - Load UGRID geometry (e.g. `final_ugrid.parquet` or `ugrid_cells_with_terrain.parquet`) with columns at least: `ID`, `lon`, `lat` (or centroid).  
   - For each UGRID cell, determine which ICON grid point(s) to use:
     - **Nearest-neighbour:** assign ICON (lat, lon) value at closest grid point to each UGRID cell.  
     - **Conservative / area-weighted (optional):** if needed for consistency, compute overlap or weights between ICON grid and UGRID cells.  
   - Result: for each (UGRID ID, 10‑min time), one `rainrate` value (float32, mm/10min).

### 3.4 Structure and format

5. **Build long-form table**  
   - Rows: one per (ID, time).  
   - Columns: `ID` (int32), `time` (datetime64), `rainrate` (float32).  
   - Same schema as `rain_kriged_*.parquet` so downstream (e.g. merger, LSTM) sees no difference.

6. **Ensemble handling**  
   - Choose one of: single member (e.g. `RAINC_01`), ensemble mean, or write multiple parquets (e.g. one per member).  
   - Document the choice in config / `rain_builder.py`.

7. **Write Parquet**  
   - Save as Parquet (e.g. `icon_rain_YYYYMMDDHH.parquet` or similar naming convention).  
   - No need to add lon/lat here if the rest of the pipeline (like the merger) adds them from UGRID when building `rain_with_discharge_*` or equivalent.

---

## 4. Summary Checklist (for implementation)

| # | Transformation | Input | Output |
|---|----------------|-------|--------|
| 1 | Cumulative → 30‑min depth | RAINC(t) | diff(RAINC) in mm/30min |
| 2 | 30‑min depth → 10‑min depth | mm/30min | mm/10min (rainrate) |
| 3 | 30‑min times → 10‑min times | 30‑min datetimes | 10‑min datetimes |
| 4 | 2D (lat, lon) → 1D (ID) | (time, lat, lon) array | (ID, time, rainrate) table |
| 5 | Schema & types | — | ID int32, time datetime64, rainrate float32 |
| 6 | Ensemble → single series | RAINC_01…20 | One series (mean or one member) |
| 7 | Write | In-memory DataFrame | Parquet |

This report and the two inspection scripts (`scripts/inspect_icon_nc.py`, `scripts/inspect_rmcomp_parquet.py`) are the reference for implementing `src/rain_builder.py` without assuming ICON and RMCOMP are the same.
