from __future__ import annotations

from pathlib import Path
import cdsapi

OUT_DIR = Path("data/raw/era5")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Ireland-ish bbox: [North, West, South, East]
AREA = [56, -12, 51, -5]

VARS = [
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
    "surface_solar_radiation_downwards",
    "2m_temperature",
]

def download_month(year: int, month: int) -> Path:
    c = cdsapi.Client()

    out_path = OUT_DIR / f"era5_{year}_{month:02d}.nc"
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"✅ Exists, skipping: {out_path}")
        return out_path

    req = {
        "product_type": "reanalysis",
        "variable": VARS,
        "year": str(year),
        "month": f"{month:02d}",
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(0, 24)],
        "area": AREA,
        "format": "netcdf",
    }

    print(f"⬇️ Downloading ERA5 {year}-{month:02d} -> {out_path}")
    c.retrieve("reanalysis-era5-single-levels", req, str(out_path))
    print(f"✅ Done: {out_path}")
    return out_path

def download_year(year: int) -> None:
    for m in range(1, 13):
        download_month(year, m)

if __name__ == "__main__":
    download_year(2024)
