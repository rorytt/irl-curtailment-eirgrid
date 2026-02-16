# src/era5/check_node_weather.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def main(year: int) -> None:
    wx_path = Path(f"data/processed/era5_node_points/era5_nodes_{year}.parquet")
    if not wx_path.exists():
        raise FileNotFoundError(f"Missing: {wx_path} (run extract_points first)")

    wx = pd.read_parquet(wx_path)
    nodes = pd.read_csv("data/raw/nodes.csv").rename(
        columns={"Node": "node_id", "Lat": "lat", "Lon": "lon", "Region": "region"}
    )

    wx["node_id"] = wx["node_id"].astype(str)
    nodes["node_id"] = nodes["node_id"].astype(str)

    # Join region + lat/lon for geography checks
    wx = wx.merge(nodes[["node_id", "region", "lat", "lon"]], on="node_id", how="left")

    # Ensure datetime is UTC
    wx["datetime"] = pd.to_datetime(wx["datetime"], utc=True)

    # Core derived variables
    wx["ws_ms"] = np.sqrt(wx["u100_ms"] ** 2 + wx["v100_ms"] ** 2)
    wx["ghi_wm2"] = wx["ssrd_jm2"] / 3600.0

    # -------------------------
    # Basic structure checks
    # -------------------------
    n_nodes = wx["node_id"].nunique()
    n_hours = wx["datetime"].nunique()
    expected_rows = n_nodes * n_hours
    dup = wx.duplicated(subset=["node_id", "datetime"]).any()
    any_nan = wx[["u100_ms", "v100_ms", "t2m_k", "ssrd_jm2"]].isna().any().any()

    print(f"=== Node weather checks: {year} ===")
    print(f"File: {wx_path}")
    print(f"Nodes: {n_nodes} | Hours: {n_hours} | Rows: {len(wx)} | Expected: {expected_rows}")
    print(f"Duplicates (node_id, datetime): {'YES' if dup else 'NO'}")
    print(f"Any NaNs in core vars: {'YES' if any_nan else 'NO'}")

    # -------------------------
    # Coastal vs inland wind check
    # -------------------------
    node_mean = wx.groupby("node_id", as_index=False)["ws_ms"].mean()
    node_mean = node_mean.merge(nodes[["node_id", "region", "lat", "lon"]], on="node_id", how="left")

    # Simple Ireland-ish heuristics:
    coastal = node_mean[(node_mean["lon"] <= -9.2) | (node_mean["lon"] >= -6.3)].copy()
    inland = node_mean[(node_mean["lon"].between(-8.6, -7.0)) & (node_mean["lat"].between(53.0, 54.7))].copy()

    print("\n--- Wind sanity: coastal vs inland (mean wind speed) ---")
    if len(coastal) == 0 or len(inland) == 0:
        print("NOTE: coastal/inland selection returned n=0 for one set. Adjust thresholds if needed.")
        print(f"coastal n={len(coastal)}, inland n={len(inland)}")
    else:
        print("Coastal nodes (top 5 by mean ws):")
        print(
            coastal.sort_values("ws_ms", ascending=False)
            .head(5)[["node_id", "region", "lat", "lon", "ws_ms"]]
            .to_string(index=False, formatters={"ws_ms": lambda x: f"{x:.2f}"})
        )
        print("\nInland nodes (top 5 by mean ws):")
        print(
            inland.sort_values("ws_ms", ascending=False)
            .head(5)[["node_id", "region", "lat", "lon", "ws_ms"]]
            .to_string(index=False, formatters={"ws_ms": lambda x: f"{x:.2f}"})
        )
        print("\nSummary:")
        print(f"Coastal mean ws: {coastal.ws_ms.mean():.2f} m/s (n={len(coastal)})")
        print(f"Inland  mean ws: {inland.ws_ms.mean():.2f} m/s (n={len(inland)})")

    # -------------------------
    # SSRD sanity: south vs north (daytime only)
    # -------------------------
    day = wx[wx["ghi_wm2"] > 0].copy()
    north_regions = ["NW", "NE"]
    south_regions = ["SW", "SE"]

    north = day[day["region"].isin(north_regions)]
    south = day[day["region"].isin(south_regions)]

    def _summary(df: pd.DataFrame, name: str) -> pd.Series:
        return pd.Series(
            {
                "set": name,
                "nodes": df["node_id"].nunique(),
                "hours": len(df),
                "mean_GHI_Wm2": df["ghi_wm2"].mean(),
                "p95_GHI_Wm2": df["ghi_wm2"].quantile(0.95) if len(df) else np.nan,
            }
        )

    print("\n--- Solar sanity: North vs South (daytime only; GHI proxy from SSRD) ---")
    out = pd.concat([_summary(north, "north (NW+NE)"), _summary(south, "south (SW+SE)")], axis=1).T
    out["mean_GHI_Wm2"] = out["mean_GHI_Wm2"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")
    out["p95_GHI_Wm2"] = out["p95_GHI_Wm2"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")
    print(out.to_string(index=False))

    # -------------------------
    # Region table (wind mean, daytime solar mean)
    # -------------------------
    wind_reg = wx.groupby("region")["ws_ms"].mean().rename("mean_ws_ms")
    solar_reg = day.groupby("region")["ghi_wm2"].mean().rename("day_mean_GHI_Wm2")

    tab = pd.concat([wind_reg, solar_reg], axis=1).reset_index().sort_values("region")
    tab["mean_ws_ms"] = tab["mean_ws_ms"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "nan")
    tab["day_mean_GHI_Wm2"] = tab["day_mean_GHI_Wm2"].map(lambda x: f"{x:.1f}" if pd.notna(x) else "nan")

    print("\n--- Region summary table ---")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks for node weather parquet (ERA5 mapped to nodes).")
    parser.add_argument("--year", type=int, required=True, help="Year to check (e.g. 2024)")
    args = parser.parse_args()
    main(args.year)
