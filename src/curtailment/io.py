# src/curtailment/io.py
from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_PROCESSED = Path("data/processed")
DATA_RAW = Path("data/raw")


def _read_parquet_yearly(pattern: str, years: list[int]) -> pd.DataFrame:
    """Read one parquet per year and concat, adding weather_year from filename/year."""
    dfs = []
    for y in years:
        p = DATA_PROCESSED / pattern.format(year=y)
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")
        df = pd.read_parquet(p)
        df["weather_year"] = y
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def load_nodes_mapping() -> pd.DataFrame:
    """
    data/raw/nodes.csv
    columns: Region,Node,Lat,Lon
    Returns: node_id, region, lat, lon
    """
    p = DATA_RAW / "nodes.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)

    # Standardise column names
    df = df.rename(
        columns={
            "Region": "region",
            "Node": "node_id",
            "Lat": "lat",
            "Lon": "lon",
        }
    )

    # Clean up types
    df["region"] = df["region"].astype(str).str.strip()
    df["node_id"] = df["node_id"].astype(str).str.strip()
    if "lat" in df.columns:
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    if "lon" in df.columns:
        df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    return df[["node_id", "region", "lat", "lon"]].drop_duplicates(subset=["node_id"])


def load_availability_nodes_calibrated(years: list[int]) -> pd.DataFrame:
    """
    Loads node-hourly calibrated availability:
      file columns: datetime, node_id, technology, available_mwh

    Merges in node->region from data/raw/nodes.csv.

    Returns:
      weather_year, datetime, node_id, region, technology, available_mwh
    """
    df = _read_parquet_yearly(
        pattern="availability_node_hourly_calibrated/availability_nodes_{year}_calibrated.parquet",
        years=years,
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df["node_id"] = df["node_id"].astype(str).str.strip()

    nodes = load_nodes_mapping()
    df = df.merge(nodes[["node_id", "region"]], on="node_id", how="left")

    # Hard fail if we couldn't map nodes -> region (this protects the model)
    unmapped = df["region"].isna().sum()
    if unmapped > 0:
        sample = df.loc[df["region"].isna(), "node_id"].drop_duplicates().head(10).tolist()
        raise ValueError(
            f"{unmapped} node-hour rows have node_id not found in data/raw/nodes.csv. "
            f"Example missing node_id: {sample}"
        )

    return df[["weather_year", "datetime", "node_id", "region", "technology", "available_mwh"]]


def load_availability_regions_calibrated(years: list[int]) -> pd.DataFrame:
    """
    Loads region-hourly calibrated availability:
      file columns: datetime, region, technology, available_mwh

    Returns:
      weather_year, datetime, region, technology, available_mwh
    """
    df = _read_parquet_yearly(
        pattern="availability_region_hourly_calibrated/availability_regions_{year}_calibrated.parquet",
        years=years,
    )
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df["region"] = df["region"].astype(str).str.strip()
    return df[["weather_year", "datetime", "region", "technology", "available_mwh"]]


def load_constraint_national_hourly(years: list[int], technology: str = "wind") -> pd.DataFrame:
    """
    data/processed/constraint_national_hourly.csv
    columns: datetime, weather_year, technology, constraint_hourly_MWh
    """
    p = DATA_PROCESSED / "constraint_national_hourly.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df["technology"] = df["technology"].astype(str).str.lower()

    df = df[df["weather_year"].isin(years)]
    df = df[df["technology"].eq(technology.lower())]
    df = df.rename(columns={"constraint_hourly_MWh": "dd_mwh"})
    return df[["weather_year", "datetime", "dd_mwh"]]


def load_recorded_availability_hourly_national(years: list[int], technology: str = "wind") -> pd.DataFrame:
    """
    data/processed/recorded_availability_hourly_national.csv
    columns: datetime, weather_year, technology, available_hourly_MWh
    """
    p = DATA_PROCESSED / "recorded_availability_hourly_national.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    df = pd.read_csv(p)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df["technology"] = df["technology"].astype(str).str.lower()

    df = df[df["weather_year"].isin(years)]
    df = df[df["technology"].eq(technology.lower())]
    df = df.rename(columns={"available_hourly_MWh": "avail_den_mwh"})
    return df[["weather_year", "datetime", "avail_den_mwh"]]


def load_constraint_percent_region_monthly(years: list[int], technology: str = "wind") -> pd.DataFrame:
    """
    data/raw/constraint_percent_region_monthly.csv
    columns: year, month, region, constraint_percent

    Returns:
      weather_year, month (month-start ts), region, dd_pct (0-1)
    """
    p = DATA_RAW / "constraint_percent_region_monthly.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")

    df = pd.read_csv(p)
    df = df.rename(columns={"year": "weather_year", "constraint_percent": "dd_pct"})
    df["region"] = df["region"].astype(str).str.strip()
    df = df[df["weather_year"].isin(years)]

    # month numeric -> month start timestamp
    df["month"] = pd.to_datetime(
        df["weather_year"].astype(str) + "-" + df["month"].astype(int).astype(str).str.zfill(2) + "-01"
    , utc=True).dt.tz_convert(None)

    # normalize % to fraction if needed
    if df["dd_pct"].max() > 1.0:
        df["dd_pct"] = df["dd_pct"] / 100.0

    _ = technology  # file is wind-only per your description
    return df[["weather_year", "month", "region", "dd_pct"]]
