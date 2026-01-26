from __future__ import annotations
from pathlib import Path
import xarray as xr
import pandas as pd

ERA5_DIR = Path("data/raw/era5")

# What we expect to find in every file (names can vary; we'll check both possibilities)
EXPECTED_VARS = {
    "u100": ["u100", "100m_u_component_of_wind"],
    "v100": ["v100", "100m_v_component_of_wind"],
    "ssrd": ["ssrd", "surface_solar_radiation_downwards"],
    "t2m":  ["t2m", "2m_temperature"],
}

def find_var(ds: xr.Dataset, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in ds.data_vars:
            return c
    return None

def expected_hours_in_month(year: int, month: int) -> int:
    start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    end = (start + pd.offsets.MonthBegin(1))
    return int((end - start) / pd.Timedelta(hours=1))

def main(year: int = 2024) -> None:
    files = sorted(ERA5_DIR.glob(f"era5_{year}_*.nc"))
    if not files:
        raise FileNotFoundError(f"No files found in {ERA5_DIR} for year={year}")

    print(f"Found {len(files)} files")
    ok = True

    for f in files:
        month = int(f.stem.split("_")[-1])
        print(f"\n--- {f.name} ---")
        print(f"Size: {f.stat().st_size/1e6:.1f} MB")

        ds = xr.open_dataset(f)

        # Check time coord
        if "time" not in ds.coords:
            print("❌ Missing time coordinate")
            ok = False
            continue

        t = pd.to_datetime(ds["time"].values)
        t = pd.DatetimeIndex(t).tz_localize("UTC") if t.tz is None else pd.DatetimeIndex(t)

        print(f"Time start: {t.min()}")
        print(f"Time end:   {t.max()}")
        print(f"Timesteps:  {len(t)}")

        # Hourly continuity check inside the file
        inferred = pd.infer_freq(t[:200]) if len(t) >= 10 else None
        print(f"Inferred freq: {inferred}")

        # Expected hour count for the month
        exp = expected_hours_in_month(year, month)
        if len(t) != exp:
            print(f"⚠️ Expected {exp} hours, got {len(t)}")
            # not always fatal, but suspicious
            ok = False

        # Check missing hours
        full = pd.date_range(t.min(), t.max(), freq="1H", tz="UTC")
        missing = full.difference(t)
        if len(missing) > 0:
            print(f"⚠️ Missing hours inside month: {len(missing)} (showing up to 5)")
            print(missing[:5])
            ok = False
        else:
            print("✅ No missing hours detected")

        # Check variables exist
        for logical, candidates in EXPECTED_VARS.items():
            vname = find_var(ds, candidates)
            if vname is None:
                print(f"❌ Missing variable for {logical}: tried {candidates}")
                ok = False
            else:
                print(f"✅ {logical} -> {vname}")

        ds.close()

    print("\n====================")
    if ok:
        print("✅ ERA5 monthly files look consistent and complete.")
    else:
        print("⚠️ ERA5 validation found issues. See messages above.")

if __name__ == "__main__":
    main(2024)
