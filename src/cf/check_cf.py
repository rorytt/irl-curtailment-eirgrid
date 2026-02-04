# src/cf/check_cf.py
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def cap_weighted_mean(node_mean: pd.DataFrame) -> float:
    """
    node_mean must have columns: cf_mean, p_rated_mw
    """
    w = node_mean["p_rated_mw"].to_numpy(dtype=float)
    x = node_mean["cf_mean"].to_numpy(dtype=float)
    denom = w.sum()
    return float((w * x).sum() / denom) if denom > 0 else float("nan")


def main(year: int) -> None:
    cf_path = Path(f"data/processed/cf_node_hourly/cf_nodes_{year}.parquet")
    if not cf_path.exists():
        raise FileNotFoundError(f"Missing: {cf_path} (run build_cf first)")

    cf = pd.read_parquet(cf_path)
    cf["datetime"] = pd.to_datetime(cf["datetime"], utc=True)
    cf["node_id"] = cf["node_id"].astype(str)
    cf["technology"] = cf["technology"].astype(str).str.strip().str.lower()

    # capacities from generators.csv
    gens = pd.read_csv("data/raw/generators.csv")
    gens["node_id"] = gens["node_id"].astype(str)
    gens["technology"] = gens["technology"].astype(str).str.strip().str.lower()

    cap = (
        gens.groupby(["node_id", "technology"], as_index=False)
        .agg(p_rated_mw=("rated_MW", "sum"))
    )

    # regions from nodes.csv (optional but helpful)
    nodes = pd.read_csv("data/raw/nodes.csv").rename(columns={"Node": "node_id", "Region": "region"})
    nodes["node_id"] = nodes["node_id"].astype(str)
    node_region = nodes[["node_id", "region"]].drop_duplicates("node_id")

    # keep only node-tech combos that actually exist in capacity table
    cf2 = cf.merge(cap, on=["node_id", "technology"], how="inner")
    cf2 = cf2.merge(node_region, on="node_id", how="left")

    print(f"=== CF checker: {year} ===")
    print(f"File: {cf_path}")
    print(f"Hours in CF file: {cf2['datetime'].nunique():,}")
    print(f"Node-tech pairs with capacity: {cf2[['node_id','technology']].drop_duplicates().shape[0]:,}")

    for tech in ["wind", "solar"]:
        t = cf2[cf2["technology"] == tech].copy()
        if len(t) == 0:
            print(f"\n--- {tech.upper()} ---\n(no rows)")
            continue

        # hourly min/max across all node-hours
        hourly_min = float(t["cf"].min())
        hourly_max = float(t["cf"].max())

        # mean CF per node (time-average)
        node_mean = (
            t.groupby(["node_id", "technology"], as_index=False)
            .agg(cf_mean=("cf", "mean"), p_rated_mw=("p_rated_mw", "first"), region=("region", "first"))
        )

        node_min = float(node_mean["cf_mean"].min())
        node_max = float(node_mean["cf_mean"].max())

        cw_mean = cap_weighted_mean(node_mean)

        print(f"\n--- {tech.upper()} ---")
        print(f"Hourly CF: min={hourly_min:.3f}, max={hourly_max:.3f}")
        print(f"Node-mean CF: min={node_min:.3f}, max={node_max:.3f}")
        print(f"Capacity-weighted mean CF: {cw_mean:.3f}")

        # top/bottom 5 nodes by mean CF
        top5 = node_mean.sort_values("cf_mean", ascending=False).head(5).copy()
        bot5 = node_mean.sort_values("cf_mean", ascending=True).head(5).copy()

        def fmt(df: pd.DataFrame) -> pd.DataFrame:
            out = df[["node_id", "region", "p_rated_mw", "cf_mean"]].copy()
            out["p_rated_mw"] = out["p_rated_mw"].map(lambda x: f"{x:.1f}")
            out["cf_mean"] = out["cf_mean"].map(lambda x: f"{x:.3f}")
            return out

        print("\nTop 5 nodes by mean CF:")
        print(fmt(top5).to_string(index=False))

        print("\nBottom 5 nodes by mean CF:")
        print(fmt(bot5).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks + rankings for node-hourly capacity factors.")
    parser.add_argument("--year", type=int, required=True, help="Year to check (e.g. 2024)")
    args = parser.parse_args()
    main(args.year)
