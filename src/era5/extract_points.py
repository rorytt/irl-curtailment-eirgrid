from __future__ import annotations
from pathlib import Path
import pandas as pd
import xarray as xr

def extract_node_weather(nodes: pd.DataFrame, era5_nc: Path) -> pd.DataFrame:
    """
    Returns long dataframe:
    datetime, node_id, u100, v100, ssrd_jm2, t2m_k
    """
    ds = xr.open_dataset(era5_nc)

    # Standardise node columns
    n = nodes.rename(columns={"Node": "node_id", "Lat": "lat", "Lon": "lon"}).copy()

    out = []
    for _, row in n.iterrows():
        sub = ds.sel(latitude=row["lat"], longitude=row["lon"], method="nearest")
        df = sub[["u100", "v100", "ssrd", "t2m"]].to_dataframe().reset_index()
        df["node_id"] = row["node_id"]
        out.append(df)

    wx = pd.concat(out, ignore_index=True)

    # Rename to explicit units
    wx = wx.rename(columns={
        "time": "datetime",
        "u100": "u100_ms",
        "v100": "v100_ms",
        "ssrd": "ssrd_jm2",
        "t2m": "t2m_k",
    })

    return wx[["datetime", "node_id", "u100_ms", "v100_ms", "ssrd_jm2", "t2m_k"]]
