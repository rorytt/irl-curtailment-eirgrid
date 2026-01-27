from __future__ import annotations

from pathlib import Path
import xarray as xr


ERA5_DIR = Path("data/raw/era5_nc")
OUT_DIR = Path("data/processed/era5_monthly")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _standardize_time(ds: xr.Dataset) -> xr.Dataset:
    # Your files use valid_time; standardize to time for downstream consistency
    if "valid_time" in ds.coords:
        ds = ds.rename({"valid_time": "time"})
    return ds


def merge_month(year: int, month: int) -> Path:
    mm = f"{month:02d}"

    instant_path = ERA5_DIR / f"era5_{year}_{mm}_instant.nc"
    accum_path = ERA5_DIR / f"era5_{year}_{mm}_accum.nc"
    out_path = OUT_DIR / f"era5_{year}_{mm}.nc"

    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    if not instant_path.exists():
        raise FileNotFoundError(instant_path)
    if not accum_path.exists():
        raise FileNotFoundError(accum_path)

    ds_i = xr.open_dataset(instant_path, engine="netcdf4")
    ds_a = xr.open_dataset(accum_path, engine="netcdf4")

    ds_i = _standardize_time(ds_i)
    ds_a = _standardize_time(ds_a)

    # Drop coords that can differ but are not useful for you
    for name in ["number", "expver"]:
        if name in ds_i.coords:
            ds_i = ds_i.drop_vars(name)
        if name in ds_a.coords:
            ds_a = ds_a.drop_vars(name)

    # Merge variables into one dataset (same dims: time/lat/lon)
    ds = xr.merge([ds_i, ds_a], compat="override", join="exact")

    # Write with simple compression
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

    ds_i.close()
    ds_a.close()
    ds.close()

    return out_path


def merge_year(year: int) -> None:
    for m in range(1, 13):
        merge_month(year, m)


if __name__ == "__main__":
    merge_year(2023)
