# src/cf/build_cf.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.cf.wind import wind_cf_from_u_v
from src.cf.solar import solar_cf_pvwatts


def _cf_sanity(cf: pd.DataFrame, n_nodes_expected: int | None = None) -> str:
    """
    Concise, repeatable sanity report:
      Row count: perfect
      Coverage: complete
      Physics: credible
      Spatial logic: sound
    Plus a few helpful stats for wind/solar.
    """
    lines = []

    # Basic sizes
    n_nodes = cf["node_id"].nunique()
    n_hours = cf["datetime"].nunique()

    # ---- Row count: perfect (per tech) ----
    wind_rows = (cf["technology"] == "wind").sum()
    solar_rows = (cf["technology"] == "solar").sum()
    expected_per_tech = n_nodes * n_hours

    row_ok = (wind_rows == expected_per_tech) and (solar_rows == expected_per_tech)
    if row_ok:
        lines.append("Row count: perfect")
    else:
        lines.append(
            f"Row count: FAILED "
            f"(wind={wind_rows}/{expected_per_tech}, solar={solar_rows}/{expected_per_tech})"
        )

    # ---- Coverage: complete (no dups, no NaNs) ----
    dup = cf.duplicated(subset=["node_id", "datetime", "technology"]).any()
    nan = cf["cf"].isna().any()
    if (not dup) and (not nan):
        lines.append("Coverage: complete")
    else:
        lines.append("Coverage: FAILED (duplicates or NaNs present)")

    # ---- Physics: credible (CF bounds) ----
    in_bounds = cf["cf"].between(0.0, 1.0).all()
    if in_bounds:
        lines.append("Physics: credible")
    else:
        lines.append("Physics: FAILED (CF outside [0,1])")

    # ---- Spatial logic: sound (node CFs not all identical) ----
    node_means = cf.groupby(["technology", "node_id"])["cf"].mean()

    wind_unique = node_means.loc["wind"].nunique() if "wind" in node_means.index.get_level_values(0) else 0
    solar_unique = node_means.loc["solar"].nunique() if "solar" in node_means.index.get_level_values(0) else 0

    spatial_ok = (wind_unique > 1) and (solar_unique > 1)
    lines.append("Spatial logic: sound" if spatial_ok else "Spatial logic: FAILED (node CFs collapse)")

    # ---- Helpful summary stats ----
    wind = cf.loc[cf["technology"] == "wind", "cf"]
    solar = cf.loc[cf["technology"] == "solar", "cf"]

    if len(wind):
        lines.append(f"Wind CF: mean={wind.mean():.3f}, zero_share={(wind == 0).mean():.3%}, max={wind.max():.3f}")
    if len(solar):
        # "night-ish" zeros should be substantial
        lines.append(f"Solar CF: mean={solar.mean():.3f}, zero_share={(solar < 1e-6).mean():.3%}, max={solar.max():.3f}")

    if n_nodes_expected is not None and n_nodes != n_nodes_expected:
        lines.append(f"WARNING: nodes={n_nodes} but expected {n_nodes_expected}")

    return "\n".join(lines)


def build_cf_year(year: int) -> Path:
    in_path = Path(f"data/processed/era5_node_points/era5_nodes_{year}.parquet")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path} (run extract_points first)")

    wx = pd.read_parquet(in_path)

    # Enforce schema assumptions
    wx["datetime"] = pd.to_datetime(wx["datetime"], utc=True)
    wx["node_id"] = wx["node_id"].astype(str)

    required_cols = {"datetime", "node_id", "u100_ms", "v100_ms", "ssrd_jm2", "t2m_k"}
    missing = required_cols - set(wx.columns)
    if missing:
        raise ValueError(f"Input parquet missing required columns {missing}. Found columns: {list(wx.columns)}")

    # Read node lat/lon from nodes.csv
    nodes = pd.read_csv("data/raw/nodes.csv").rename(columns={"Node": "node_id", "Lat": "lat", "Lon": "lon"})
    nodes["node_id"] = nodes["node_id"].astype(str)
    node_latlon = nodes.set_index("node_id")[["lat", "lon"]].to_dict("index")

    # ---- WIND CF (all node-hours) ----
    wind_df = wx[["datetime", "node_id", "u100_ms", "v100_ms"]].copy()
    wind_df["cf"] = wind_cf_from_u_v(wind_df)
    wind_df["technology"] = "wind"
    wind_df = wind_df[["datetime", "node_id", "technology", "cf"]]

    # ---- SOLAR CF (upgraded PVWatts approach) ----
    solar_pieces = []
    for node_id, grp in wx.groupby("node_id", sort=False):
        if node_id not in node_latlon:
            continue
        lat = float(node_latlon[node_id]["lat"])
        lon = float(node_latlon[node_id]["lon"])

        g = grp[["datetime", "node_id", "ssrd_jm2", "t2m_k", "u100_ms", "v100_ms"]].sort_values("datetime").copy()
        g["cf"] = solar_cf_pvwatts(g, lat, lon)
        g["technology"] = "solar"
        solar_pieces.append(g[["datetime", "node_id", "technology", "cf"]])

    solar_df = pd.concat(solar_pieces, ignore_index=True)

    # Combine + sort
    cf = pd.concat([wind_df, solar_df], ignore_index=True)
    cf = cf.sort_values(["node_id", "technology", "datetime"]).reset_index(drop=True)

    # Write output
    out_dir = Path("data/processed/cf_node_hourly")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cf_nodes_{year}.parquet"

    cf.to_parquet(out_path, index=False)

    print(f"Wrote {out_path}")
    print(_cf_sanity(cf, n_nodes_expected=wx["node_id"].nunique()))

    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build node-hourly capacity factors (wind + solar) from ERA5 node points.")
    parser.add_argument("--year", type=int, default=2024, help="Year to process (default: 2024)")
    args = parser.parse_args()

    build_cf_year(args.year)
