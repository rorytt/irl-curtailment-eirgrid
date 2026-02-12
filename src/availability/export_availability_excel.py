# src/availability/export_availability_excel.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


REGION_ORDER = ["NW", "NE", "W", "SW", "SE", "MID"]  # adjust if needed


def read_region_availability(year: int, calibrated: bool) -> pd.DataFrame:
    if calibrated:
        p = Path(f"data/processed/availability_region_hourly_calibrated/availability_regions_{year}_calibrated.parquet")
    else:
        p = Path(f"data/processed/availability_region_hourly/availability_regions_{year}.parquet")

    if not p.exists():
        raise FileNotFoundError(f"Missing availability parquet: {p}")

    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()
    df["region"] = df["region"].astype(str).str.strip()
    df["year"] = year
    return df[["datetime", "year", "region", "technology", "available_mwh"]]


def add_hour_of_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds hour_of_year per (year) based on sorted datetime.
    Works for leap and non-leap years; hour_of_year will be 1..8760/8784 etc.
    """
    out = df.copy()
    out = out.sort_values(["year", "datetime"])
    out["hour_of_year"] = out.groupby("year").cumcount() + 1
    return out


def make_region_tech_pivot(df_all: pd.DataFrame, region: str, tech: str) -> pd.DataFrame:
    """
    Returns wide table for Excel plotting:
      index = hour_of_year
      columns = years
      values = available_mwh
    """
    sub = df_all[(df_all["region"] == region) & (df_all["technology"] == tech)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = add_hour_of_year(sub)

    piv = sub.pivot_table(
        index="hour_of_year",
        columns="year",
        values="available_mwh",
        aggfunc="sum",
    ).sort_index()

    # Make years columns ordered
    piv = piv.reindex(sorted(piv.columns), axis=1)
    piv.index.name = "hour_of_year"
    return piv


def main(years: Iterable[int], calibrated: bool, out_path: Path) -> None:
    years = list(years)
    years = sorted(set(years))

    dfs = [read_region_availability(y, calibrated=calibrated) for y in years]
    df_all = pd.concat(dfs, ignore_index=True)

    # Order regions if you use the standard set
    regions = [r for r in REGION_ORDER if r in df_all["region"].unique().tolist()]
    # Add any extra regions not in REGION_ORDER
    regions += [r for r in sorted(df_all["region"].unique()) if r not in regions]

    techs = [t for t in ["wind", "solar"] if t in df_all["technology"].unique().tolist()]
    techs += [t for t in sorted(df_all["technology"].unique()) if t not in techs]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Excel cannot store tz-aware datetimes
    df_all = df_all.copy()
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], utc=True).dt.tz_convert(None)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # Sheet 1: long format (best for pivot tables in Excel)
        df_all.sort_values(["year", "datetime", "region", "technology"]).to_excel(
            writer, sheet_name="long", index=False
        )

        # Sheet 2: simple system totals by region/tech/year (handy)
        monthly = df_all.copy()
        monthly["month"] = monthly["datetime"].dt.month
        monthly_sum = (
            monthly.groupby(["year", "month", "region", "technology"], as_index=False)
            .agg(total_gwh=("available_mwh", lambda s: s.sum() / 1e3))
        )
        monthly_sum.to_excel(writer, sheet_name="monthly_gwh", index=False)

        # Per region-tech: wide by year for easy overlay plots
        for region in regions:
            for tech in techs:
                piv = make_region_tech_pivot(df_all, region=region, tech=tech)
                if piv.empty:
                    continue
                sheet = f"{region}_{tech}"[:31]  # Excel sheet name limit
                piv.to_excel(writer, sheet_name=sheet)

    print(f"Wrote Excel: {out_path}")
    print(f"Years: {years}")
    print(f"Using calibrated={calibrated}")
    print("Sheets: long, monthly_gwh, plus one sheet per region_tech (wide by year for plotting).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export hourly regional availability to an Excel workbook for plotting.")
    parser.add_argument(
        "--years",
        type=str,
        required=True,
        help="Comma-separated years to export (e.g. 2021,2022,2023,2024,2025). 2021 will be included if missing.",
    )
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Use calibrated availability files (availability_region_hourly_calibrated). Default is uncalibrated.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed/exports/hourly_availability_by_region.xlsx",
        help="Output Excel path.",
    )
    args = parser.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    main(years=years, calibrated=args.calibrated, out_path=Path(args.out))
