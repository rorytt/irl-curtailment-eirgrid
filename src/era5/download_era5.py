from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import zipfile

import cdsapi


# -------------------------
# Paths
# -------------------------
ERA5_ZIP_DIR = Path("data/raw/era5_zip")
ERA5_NC_DIR = Path("data/raw/era5_nc")

ERA5_ZIP_DIR.mkdir(parents=True, exist_ok=True)
ERA5_NC_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Request config
# -------------------------
AREA = [56, -12, 51, -5]  # [North, West, South, East]

VARS = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "2m_temperature",
]

DATASET = "reanalysis-era5-single-levels"


# -------------------------
# Helpers
# -------------------------
def output_paths(year: int, month: int) -> dict[str, Path]:
    mm = f"{month:02d}"
    return {
        "instant": ERA5_NC_DIR / f"era5_{year}_{mm}_instant.nc",
        "accum": ERA5_NC_DIR / f"era5_{year}_{mm}_accum.nc",
    }


def download_month(year: int, month: int) -> Path:
    mm = f"{month:02d}"
    zip_path = ERA5_ZIP_DIR / f"era5_{year}_{mm}.zip"

    if zip_path.exists() and zip_path.stat().st_size > 0:
        return zip_path

    req = {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": str(year),
        "month": mm,
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "format": "netcdf",
    }

    c = cdsapi.Client()
    c.retrieve(DATASET, req, str(zip_path))
    return zip_path


def extract_month(zip_path: Path, year: int, month: int) -> None:
    outs = output_paths(year, month)

    if outs["instant"].exists() and outs["accum"].exists():
        return

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a zip file: {zip_path}")

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmpdir)

        nc_files = list(tmpdir.rglob("*.nc"))
        if not nc_files:
            raise RuntimeError(f"No NetCDF files in {zip_path}")

        instant = None
        accum = None

        for f in nc_files:
            name = f.name.lower()
            if "steptype-instant" in name or name.endswith("instant.nc"):
                instant = f
            elif "steptype-accum" in name or name.endswith("accum.nc"):
                accum = f

        if instant is None or accum is None:
            raise RuntimeError(f"Could not identify instant/accum files in {zip_path}")

        shutil.move(str(instant), str(outs["instant"]))
        shutil.move(str(accum), str(outs["accum"]))


def download_year(year: int) -> None:
    for month in range(1, 13):
        zip_path = download_month(year, month)
        extract_month(zip_path, year, month)


if __name__ == "__main__":
    download_year(2023)
