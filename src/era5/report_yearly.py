from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_YEARLY_DIR = Path("data/processed/era5_yearly")
DEFAULT_REPORT_DIR = Path("reports/era5")

TIME_CANDIDATES = ["time", "valid_time"]

EXPECTED_VARS = ["u100", "v100", "t2m", "ssrd"]


@dataclass
class VarStats:
    name: str
    units: str | None
    vmin: float | None
    vmax: float | None
    mean: float | None
    nan_count: int
    total_count: int


def _find_time_name(ds: xr.Dataset) -> str:
    for n in TIME_CANDIDATES:
        if n in ds.coords:
            return n
    raise ValueError(f"No time coordinate found. Tried: {TIME_CANDIDATES}")


def _expected_hours_in_year(year: int) -> int:
    start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    end = pd.Timestamp(year=year + 1, month=1, day=1, tz="UTC")
    return int((end - start) / pd.Timedelta(hours=1))


def _as_utc_index(time_values: np.ndarray) -> pd.DatetimeIndex:
    t = pd.to_datetime(time_values)
    t = pd.DatetimeIndex(t)
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def _safe_float(x) -> float | None:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except Exception:
        return None


def _var_stats(da: xr.DataArray, name: str) -> VarStats:
    units = da.attrs.get("units")
    arr = da.values

    total = int(np.prod(arr.shape))
    nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0

    vmin = _safe_float(np.nanmin(arr)) if np.issubdtype(arr.dtype, np.floating) else None
    vmax = _safe_float(np.nanmax(arr)) if np.issubdtype(arr.dtype, np.floating) else None
    mean = _safe_float(np.nanmean(arr)) if np.issubdtype(arr.dtype, np.floating) else None

    return VarStats(
        name=name,
        units=units,
        vmin=vmin,
        vmax=vmax,
        mean=mean,
        nan_count=nan_count,
        total_count=total,
    )


def _fmt_stats(s: VarStats) -> str:
    u = s.units if s.units is not None else "(units unknown)"
    return (
        f"{s.name}: units={u}, min={s.vmin}, max={s.vmax}, mean={s.mean}, "
        f"NaNs={s.nan_count}/{s.total_count}"
    )


def report_yearly(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)

    lines: list[str] = []
    lines.append("ERA5 YEARLY FILE REPORT")
    lines.append(f"Path: {path}")
    lines.append(f"Size (MB): {path.stat().st_size / 1e6:.2f}")
    lines.append("")

    ds = xr.open_dataset(path, engine="netcdf4")

    # Basic structure
    lines.append("STRUCTURE")
    lines.append(f"Dimensions: {dict(ds.dims)}")
    lines.append(f"Coordinates: {list(ds.coords)}")
    lines.append(f"Data variables: {list(ds.data_vars)}")
    lines.append("")

    # Time checks
    time_name = _find_time_name(ds)
    t = _as_utc_index(ds[time_name].values)

    # Infer year from timestamps
    years_present = sorted(set(t.year))
    lines.append("TIME CHECKS")
    lines.append(f"Time coordinate name: {time_name}")
    lines.append(f"Start: {t.min()}")
    lines.append(f"End:   {t.max()}")
    lines.append(f"Count: {len(t)}")
    lines.append(f"Years present: {years_present}")
    lines.append(f"Frequency guess (first 200): {pd.infer_freq(t[:200])}")
    lines.append("")

    # Expected hourly coverage if file is a single year
    time_ok = True
    if len(years_present) == 1:
        y = years_present[0]
        exp = _expected_hours_in_year(y)
        if len(t) != exp:
            time_ok = False
            lines.append(f"Time count mismatch: expected {exp} hours for {y}, got {len(t)}")
        else:
            lines.append(f"Expected hours for {y}: {exp} (matches)")

        full = pd.date_range(t.min(), t.max(), freq="1H", tz="UTC")
        missing = full.difference(t)
        if len(missing) > 0:
            time_ok = False
            lines.append(f"Missing hourly timestamps: {len(missing)} (showing up to 10)")
            lines.extend([f"  {x}" for x in missing[:10]])
        else:
            lines.append("Missing hourly timestamps: 0")
    else:
        lines.append("Note: file spans multiple years; skipping single-year expected-hour check.")

    lines.append("")

    # Variable presence
    lines.append("VARIABLE PRESENCE")
    missing_vars = [v for v in EXPECTED_VARS if v not in ds.data_vars]
    if missing_vars:
        lines.append(f"Missing expected variables: {missing_vars}")
    else:
        lines.append("All expected variables present: u100, v100, t2m, ssrd")
    lines.append("")

    # Variable stats
    lines.append("VARIABLE STATS")
    var_stats: dict[str, VarStats] = {}
    for v in EXPECTED_VARS:
        if v in ds.data_vars:
            s = _var_stats(ds[v], v)
            var_stats[v] = s
            lines.append(_fmt_stats(s))
    lines.append("")

    # Derived checks
    derived_ok = True
    lines.append("DERIVED CHECKS")

    if "u100" in ds.data_vars and "v100" in ds.data_vars:
        ws = (ds["u100"] ** 2 + ds["v100"] ** 2) ** 0.5
        s_ws = _var_stats(ws, "ws100")
        lines.append(_fmt_stats(s_ws))

        if s_ws.vmin is not None and s_ws.vmin < 0:
            derived_ok = False
            lines.append("ws100 sanity: FAIL (wind speed should not be negative)")
        else:
            lines.append("ws100 sanity: PASS (non-negative)")
    else:
        derived_ok = False
        lines.append("ws100: cannot compute (u100/v100 missing)")

    if "ssrd" in ds.data_vars:
        ghi = ds["ssrd"] / 3600.0
        s_ghi = _var_stats(ghi, "ghi_wm2")
        lines.append(_fmt_stats(s_ghi))

        # Physical sanity checks for ssrd/ghi
        ssrd_min = var_stats["ssrd"].vmin if "ssrd" in var_stats else None
        if ssrd_min is not None and ssrd_min < 0:
            derived_ok = False
            lines.append("ssrd sanity: FAIL (ssrd should be non-negative)")
        else:
            lines.append("ssrd sanity: PASS (non-negative min)")

        # Quick diurnal check at a representative point (middle of grid)
        lat_i = ds.dims.get("latitude", 0) // 2 if "latitude" in ds.dims else 0
        lon_i = ds.dims.get("longitude", 0) // 2 if "longitude" in ds.dims else 0
        if "latitude" in ds.dims and "longitude" in ds.dims:
            ghi_pt = ghi.isel(latitude=lat_i, longitude=lon_i).values
            t_hours = t.hour.values

            # Night proxy: hours 0-4 and 21-23 should be frequently low
            night_mask = (t_hours <= 4) | (t_hours >= 21)
            night_vals = ghi_pt[night_mask]
            if night_vals.size > 0:
                frac_near_zero = float(np.mean(night_vals < 5.0))  # 5 W/m2 threshold
                lines.append(
                    f"GHI point diurnal check (center cell): fraction of night hours < 5 W/m2 = {frac_near_zero:.3f}"
                )
            else:
                lines.append("GHI point diurnal check: could not compute (no night samples?)")
        else:
            lines.append("GHI point diurnal check: skipped (latitude/longitude dims not found)")
    else:
        derived_ok = False
        lines.append("ghi_wm2: cannot compute (ssrd missing)")

    lines.append("")

    # Summary
    lines.append("SUMMARY")
    lines.append(f"Time checks: {'PASS' if time_ok else 'FAIL'}")
    lines.append(f"Derived checks: {'PASS' if derived_ok else 'FAIL'}")
    lines.append("If all checks pass and ranges look plausible for Ireland, the file is ready for CF calculations.")

    ds.close()
    return "\n".join(lines)


def _default_path_for_year(year: int) -> Path:
    return DEFAULT_YEARLY_DIR / f"era5_{year}.nc"


def main() -> None:
    parser = argparse.ArgumentParser(description="ERA5 yearly NetCDF report")
    parser.add_argument("--year", type=int, default=None, help="Year (uses default path data/processed/era5_yearly/era5_YEAR.nc)")
    parser.add_argument("--path", type=str, default=None, help="Explicit path to a yearly NetCDF file")
    parser.add_argument("--save", action="store_true", help="Save report to reports/era5/era5_YEAR_report.txt (or based on filename)")
    args = parser.parse_args()

    if args.path is None and args.year is None:
        raise SystemExit("Provide --year YYYY or --path PATH")

    if args.path is not None:
        path = Path(args.path)
    else:
        path = _default_path_for_year(int(args.year))

    rep = report_yearly(path)
    print(rep)

    if args.save:
        DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
        if args.year is not None:
            out = DEFAULT_REPORT_DIR / f"era5_{args.year}_report.txt"
        else:
            out = DEFAULT_REPORT_DIR / f"{path.stem}_report.txt"
        out.write_text(rep)
        print(f"\nSaved report to: {out}")


if __name__ == "__main__":
    main()
