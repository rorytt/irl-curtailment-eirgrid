# src/availability/build_availability.py
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd


def _availability_sanity(av_node: pd.DataFrame) -> str:
    lines = []

    n_hours = av_node["datetime"].nunique()
    n_pairs = av_node[["node_id", "technology"]].drop_duplicates().shape[0]
    expected = n_hours * n_pairs

    if len(av_node) == expected:
        lines.append("Row count: perfect")
    else:
        lines.append(f"Row count: FAILED (got {len(av_node)}, expected {expected})")

    dup = av_node.duplicated(subset=["node_id", "datetime", "technology"]).any()
    nan = av_node[["available_mwh"]].isna().any().any()
    if (not dup) and (not nan):
        lines.append("Coverage: complete")
    else:
        lines.append("Coverage: FAILED (duplicates or NaNs present)")

    nonneg = (av_node["available_mwh"] >= -1e-9).all()
    if nonneg:
        lines.append("Physics: credible")
    else:
        lines.append("Physics: FAILED (negative availability found)")

    # quick tech stats
    bytech = av_node.groupby("technology")["available_mwh"].agg(["mean", "max"])
    for tech, row in bytech.iterrows():
        lines.append(f"{tech.title()} availability: mean={row['mean']:.3f} MWh, max={row['max']:.3f} MWh")

    # optional: show how many pairs per tech
    pairs_by_tech = av_node[["node_id","technology"]].drop_duplicates().groupby("technology").size()
    for tech, cnt in pairs_by_tech.items():
        lines.append(f"{tech.title()} node-tech pairs: {int(cnt)}")

    return "\n".join(lines)



def build_availability_year(year: int) -> None:
    cf_path = Path(f"data/processed/cf_node_hourly/cf_nodes_{year}.parquet")
    if not cf_path.exists():
        raise FileNotFoundError(f"Missing CF file: {cf_path} (run build_cf first)")

    gens = pd.read_csv("data/raw/generators.csv")
    nodes = pd.read_csv("data/raw/nodes.csv").rename(columns={"Node": "node_id", "Region": "region"})

    # standardise ids
    gens["node_id"] = gens["node_id"].astype(str)
    nodes["node_id"] = nodes["node_id"].astype(str)

    # standardise technology labels
    gens["technology"] = gens["technology"].astype(str).str.strip().str.lower()

    # node-tech capacity (MW)
    cap = (
        gens.groupby(["node_id", "technology"], as_index=False)
        .agg(p_rated_mw=("rated_MW", "sum"))
    )

    cf = pd.read_parquet(cf_path)
    cf["node_id"] = cf["node_id"].astype(str)
    cf["technology"] = cf["technology"].astype(str).str.strip().str.lower()
    cf["datetime"] = pd.to_datetime(cf["datetime"], utc=True)

    # join capacity; availability in MWh for 1-hour step
    av = cf.merge(cap, on=["node_id", "technology"], how="left")

    # if a node has no capacity for a tech, drop it (keeps outputs clean)
    av = av.dropna(subset=["p_rated_mw"]).copy()

    av["available_mwh"] = av["p_rated_mw"] * av["cf"]  # 1h timestep

    av_node = av[["datetime", "node_id", "technology", "available_mwh"]].copy()

    # ---- write node availability ----
    out_node_dir = Path("data/processed/availability_node_hourly")
    out_node_dir.mkdir(parents=True, exist_ok=True)
    out_node_path = out_node_dir / f"availability_nodes_{year}.parquet"
    av_node.to_parquet(out_node_path, index=False)

    # ---- region aggregation ----
    node_region = nodes[["node_id", "region"]].drop_duplicates("node_id")
    av_region = av_node.merge(node_region, on="node_id", how="left")

    av_region = (
        av_region.groupby(["datetime", "region", "technology"], as_index=False)
        .agg(available_mwh=("available_mwh", "sum"))
    )

    out_reg_dir = Path("data/processed/availability_region_hourly")
    out_reg_dir.mkdir(parents=True, exist_ok=True)
    out_reg_path = out_reg_dir / f"availability_regions_{year}.parquet"
    av_region.to_parquet(out_reg_path, index=False)

    print(f"Wrote {out_node_path}")
    print(f"Wrote {out_reg_path}")
    print(_availability_sanity(av_node))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build node + region availability (MWh) from CF + generator capacities.")
    parser.add_argument("--year", type=int, required=True, help="Year to process (e.g. 2024)")
    args = parser.parse_args()

    build_availability_year(args.year)
