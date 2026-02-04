from __future__ import annotations

from pathlib import Path
from re import sub
import pandas as pd
import xarray as xr
import numpy as np


def extract_node_weather(nodes: pd.DataFrame, era5_nc: Path) -> pd.DataFrame:
    """
    Returns long dataframe:
    datetime, node_id, u100_ms, v100_ms, ssrd_jm2, t2m_k
    """
    ds = xr.open_dataset(era5_nc)

    # nodes.csv expected columns: Region, Node, Lat, Lon
    n = nodes.rename(columns={"Node": "node_id", "Lat": "lat", "Lon": "lon"}).copy()

    out = []
    for _, row in n.iterrows():
        sub = ds.sel(latitude=row["lat"], longitude=row["lon"], method="nearest")
        df = sub[["u100", "v100", "ssrd", "t2m"]].to_dataframe().reset_index()
        # capture the actual ERA5 grid point chosen
        era5_lat = float(sub["latitude"].values)
        era5_lon = float(sub["longitude"].values)

        df = sub[["u100", "v100", "ssrd", "t2m"]].to_dataframe().reset_index()
        df["node_id"] = str(row["node_id"])

        # store mapping metadata (repeated per hour, but we'll dedupe in sanity check)
        df["node_lat"] = float(row["lat"])
        df["node_lon"] = float(row["lon"])
        df["era5_lat"] = era5_lat
        df["era5_lon"] = era5_lon

        out.append(df)

    wx = pd.concat(out, ignore_index=True)

    wx = wx.rename(
        columns={
            "time": "datetime",
            "u100": "u100_ms",
            "v100": "v100_ms",
            "ssrd": "ssrd_jm2",
            "t2m": "t2m_k",
        }
    )

    # ERA5 is effectively UTC
    wx["datetime"] = pd.to_datetime(wx["datetime"], utc=True)

    return wx[[
        "datetime", "node_id",
        "u100_ms", "v100_ms", "ssrd_jm2", "t2m_k",
        "node_lat", "node_lon", "era5_lat", "era5_lon"
    ]]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def mapping_distance_summary(wx: pd.DataFrame) -> tuple[float, float, float, int]:
    # one row per node
    m = (wx[["node_id","node_lat","node_lon","era5_lat","era5_lon"]]
         .drop_duplicates("node_id")
         .copy())

    dist_km = haversine_km(m["node_lat"], m["node_lon"], m["era5_lat"], m["era5_lon"])
    m["dist_km"] = dist_km

    # count how many unique ERA5 gridpoints used
    n_gridpoints = m.drop_duplicates(["era5_lat","era5_lon"]).shape[0]

    return float(m["dist_km"].min()), float(m["dist_km"].mean()), float(m["dist_km"].max()), int(n_gridpoints)

def sanity_check(wx: pd.DataFrame) -> str:
    lines = []

    n_nodes = wx["node_id"].nunique()
    times = wx["datetime"].sort_values().unique()
    n_hours = len(times)
    expected_rows = n_nodes * n_hours

    # 1) Row count
    if len(wx) == expected_rows:
        lines.append("Row count: perfect")
    else:
        lines.append(f"Row count: FAILED (got {len(wx)}, expected {expected_rows})")

    # 2) Coverage
    dup = wx.duplicated(subset=["node_id", "datetime"]).any()
    missing = wx.isna().any().any()
    if not dup and not missing:
        lines.append("Coverage: complete")
    else:
        lines.append("Coverage: FAILED (duplicates or NaNs present)")

    # 3) Physics
    wind_speed = (wx["u100_ms"]**2 + wx["v100_ms"]**2) ** 0.5
    phys_ok = (
        (wind_speed >= 0).all()
        and (wx["ssrd_jm2"] >= 0).all()
        and (wx["t2m_k"].between(223, 323)).all()  # −50°C to +50°C
    )

    night_zero = (wx.loc[wx["ssrd_jm2"] == 0].shape[0] > 0)

    if phys_ok and night_zero:
        lines.append("Physics: credible")
    else:
        lines.append("Physics: FAILED (non-physical values)")

    # 4) Spatial logic
    sample = wx.groupby("node_id")["u100_ms"].mean()
    if sample.nunique() > 1:
        lines.append("Spatial logic: sound")
    else:
        lines.append("Spatial logic: FAILED (nodes collapse to identical series)")
    
    # 5) Mapping distances
    dmin, dmean, dmax, ngrid = mapping_distance_summary(wx)
    lines.append(f"Mapping dist (km): min={dmin:.2f}, mean={dmean:.2f}, max={dmax:.2f}")
    lines.append(f"Mapped ERA5 gridpoints: {ngrid} (nodes={wx['node_id'].nunique()})")

    return "\n".join(lines)

if __name__ == "__main__":
    year = 2021

    nodes = pd.read_csv("data/raw/nodes.csv")
    era5_path = Path(f"data/processed/era5_yearly/era5_{year}.nc")

    wx = extract_node_weather(nodes, era5_path)

    out_dir = Path("data/processed/era5_node_points")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"era5_nodes_{year}.parquet"

    # ---- sanity check (prints to console) ----
    print(sanity_check(wx))

    # ---- write one-row-per-node mapping file ----
    mapping = (
        wx[["node_id","node_lat","node_lon","era5_lat","era5_lon"]]
        .drop_duplicates("node_id")
        .sort_values("node_id")
    )
    mapping.to_csv(out_dir / f"node_to_era5_grid_{year}.csv", index=False)

    # ---- drop mapping cols before saving parquet ----
    wx_out = wx.drop(columns=["node_lat","node_lon","era5_lat","era5_lon"])
    wx_out.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")
   

