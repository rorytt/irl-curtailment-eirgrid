from __future__ import annotations

from pathlib import Path
import xarray as xr


IN_DIR = Path("data/processed/era5_monthly")
OUT_DIR = Path("data/processed/era5_yearly")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_year(year: int) -> Path:
    files = sorted(IN_DIR.glob(f"era5_{year}_??.nc"))
    if len(files) != 12:
        raise FileNotFoundError(
            f"Expected 12 monthly files in {IN_DIR} for {year}, found {len(files)}"
        )

    out_path = OUT_DIR / f"era5_{year}.nc"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    dsets: list[xr.Dataset] = []
    try:
        for f in files:
            ds = xr.open_dataset(f, engine="netcdf4")
            dsets.append(ds)

        # Concatenate along time and ensure chronological ordering
        year_ds = xr.concat(dsets, dim="time")
        year_ds = year_ds.sortby("time")

        # Basic compression
        encoding = {v: {"zlib": True, "complevel": 4} for v in year_ds.data_vars}
        year_ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)

        year_ds.close()
    finally:
        for ds in dsets:
            try:
                ds.close()
            except Exception:
                pass

    return out_path


if __name__ == "__main__":
    build_year(2023)
