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


def read_node_availability(year: int, calibrated: bool) -> pd.DataFrame:
    if calibrated:
        p = Path(f"data/processed/availability_node_hourly_calibrated/availability_nodes_{year}_calibrated.parquet")
    else:
        p = Path(f"data/processed/availability_node_hourly/availability_nodes_{year}.parquet")

    if not p.exists():
        raise FileNotFoundError(f"Missing availability parquet: {p}")

    df = pd.read_parquet(p)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()
    df["node_id"] = df["node_id"].astype(str).str.strip()
    df["year"] = year
    return df[["datetime", "year", "node_id", "technology", "available_mwh"]]


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


def make_node_tech_pivot(df_all: pd.DataFrame, node: str, tech: str) -> pd.DataFrame:
    """
    Returns wide table for Excel plotting:
      index = hour_of_year
      columns = years
      values = available_mwh
    """
    sub = df_all[(df_all["node_id"] == node) & (df_all["technology"] == tech)].copy()
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


def main(years: Iterable[int], calibrated: bool, out_path: Path, export_nodes: bool = False, node_list: list[str] | None = None) -> None:
    years = list(years)
    years = sorted(set(years))

    # Export regional availability
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

    # Export nodal availability if requested
    if export_nodes:
        node_dfs = [read_node_availability(y, calibrated=calibrated) for y in years]
        node_df_all = pd.concat(node_dfs, ignore_index=True)

        node_techs = [t for t in ["wind", "solar"] if t in node_df_all["technology"].unique().tolist()]
        node_techs += [t for t in sorted(node_df_all["technology"].unique()) if t not in node_techs]

        # Get all available nodes
        all_nodes = sorted(node_df_all["node_id"].unique().tolist())
        
        # Filter to specific nodes if requested, otherwise use all
        if node_list:
            nodes = [n for n in node_list if n in all_nodes]
            if not nodes:
                print(f"Warning: None of the requested nodes {node_list} found in data. Available nodes: {all_nodes}")
                nodes = all_nodes
        else:
            nodes = all_nodes

        # Generate nodal output path by replacing "region" with "node" in the filename
        node_filename = out_path.name.replace("region", "node")
        node_out_path = out_path.parent / node_filename

        # Excel cannot store tz-aware datetimes
        node_df_all = node_df_all.copy()
        node_df_all["datetime"] = pd.to_datetime(node_df_all["datetime"], utc=True).dt.tz_convert(None)

        # Filter data to selected nodes
        node_df_filtered = node_df_all[node_df_all["node_id"].isin(nodes)].copy()

        with pd.ExcelWriter(node_out_path, engine="openpyxl") as writer:
            # Sheet 1: long format
            node_df_filtered.sort_values(["year", "datetime", "node_id", "technology"]).to_excel(
                writer, sheet_name="long", index=False
            )

            # Sheet 2: simple system totals by node/tech/year
            node_monthly = node_df_filtered.copy()
            node_monthly["month"] = node_monthly["datetime"].dt.month
            node_monthly_sum = (
                node_monthly.groupby(["year", "month", "node_id", "technology"], as_index=False)
                .agg(total_gwh=("available_mwh", lambda s: s.sum() / 1e3))
            )
            node_monthly_sum.to_excel(writer, sheet_name="monthly_gwh", index=False)

            # Per node-tech: wide by year for easy overlay plots
            for node in nodes:
                for tech in node_techs:
                    piv = make_node_tech_pivot(node_df_filtered, node=node, tech=tech)
                    if piv.empty:
                        continue
                    sheet = f"{node}_{tech}"[:31]  # Excel sheet name limit
                    piv.to_excel(writer, sheet_name=sheet)

        print(f"\nWrote Excel (nodal): {node_out_path}")
        print(f"Years: {years}")
        print(f"Nodes exported: {nodes}")
        print(f"Using calibrated={calibrated}")
        print("Sheets: long, monthly_gwh, plus one sheet per node_tech (wide by year for plotting).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export hourly regional/nodal availability to Excel workbooks for plotting.")
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
        help="Output Excel path for regional data.",
    )
    parser.add_argument(
        "--nodal",
        action="store_true",
        help="Also export nodal availability data to a separate Excel file.",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        default=None,
        help="Comma-separated list of specific node IDs to export (e.g. Ardnacrusha,Castlerea). If not specified, all nodes will be exported.",
    )
    args = parser.parse_args()

    years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
    node_list = [n.strip() for n in args.nodes.split(",")] if args.nodes else None
    main(years=years, calibrated=args.calibrated, out_path=Path(args.out), export_nodes=args.nodal, node_list=node_list)
