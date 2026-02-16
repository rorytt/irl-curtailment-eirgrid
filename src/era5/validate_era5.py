from __future__ import annotations

from pathlib import Path
import pandas as pd
import xarray as xr


ERA5_DIR = Path("data/raw/era5_nc")

ALIASES = {
    "u100": ["u100", "100m_u_component_of_wind"],
    "v100": ["v100", "100m_v_component_of_wind"],
    "t2m": ["t2m", "2m_temperature"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
}

EXPECTED_BY_TYPE = {
    "instant": ["u100", "v100", "t2m"],
    "accum": ["ssrd"],
}

TIME_CANDIDATES = ["valid_time", "time"]


def find_var(ds: xr.Dataset, logical: str) -> str | None:
    for name in ALIASES.get(logical, []):
        if name in ds.data_vars:
            return name
    return None


def find_time_coord(ds: xr.Dataset) -> str | None:
    for name in TIME_CANDIDATES:
        if name in ds.coords:
            return name
    return None


def expected_hours_in_month(year: int, month: int) -> int:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    return int((end - start) / pd.Timedelta(hours=1))


def validate_one(path: Path, year: int, month: int, ftype: str) -> list[str]:
    issues: list[str] = []

    try:
        ds = xr.open_dataset(path, engine="netcdf4")
    except Exception as e:
        return [f"could not open dataset: {e}"]

    tname = find_time_coord(ds)
    if tname is None:
        ds.close()
        return ["missing time coordinate"]

    t = pd.to_datetime(ds[tname].values)
    t = pd.DatetimeIndex(t)
    if t.tz is None:
        t = t.tz_localize("UTC")

    exp = expected_hours_in_month(year, month)
    if len(t) != exp:
        issues.append(f"expected {exp} hours, got {len(t)}")

    full = pd.date_range(t.min(), t.max(), freq="1h", tz="UTC")
    missing = full.difference(t)
    if len(missing) > 0:
        issues.append(f"missing {len(missing)} hourly timestamps")

    for logical in EXPECTED_BY_TYPE[ftype]:
        if find_var(ds, logical) is None:
            issues.append(f"missing variable: {logical}")

    ds.close()
    return issues


def main(year: int = 2024) -> None:
    overall_ok = True

    for month in range(1, 13):
        mm = f"{month:02d}"
        instant = ERA5_DIR / f"era5_{year}_{mm}_instant.nc"
        accum = ERA5_DIR / f"era5_{year}_{mm}_accum.nc"

        month_issues: list[tuple[str, list[str]]] = []

        if not instant.exists():
            month_issues.append(("instant", ["file missing"]))
        else:
            issues = validate_one(instant, year, month, "instant")
            if issues:
                month_issues.append(("instant", issues))

        if not accum.exists():
            month_issues.append(("accum", ["file missing"]))
        else:
            issues = validate_one(accum, year, month, "accum")
            if issues:
                month_issues.append(("accum", issues))

        if month_issues:
            overall_ok = False
            print(f"{year}-{mm}: FAIL")
            for label, issues in month_issues:
                for i in issues:
                    print(f"  - {label}: {i}")
        else:
            print(f"{year}-{mm}: PASS")

    if not overall_ok:
        raise RuntimeError("ERA5 validation failed")


if __name__ == "__main__":
    main(2021)
