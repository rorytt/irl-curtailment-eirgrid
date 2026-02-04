# src/availability/check_availability.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def _basic_checks_node(av: pd.DataFrame, cap: pd.DataFrame) -> list[str]:
    lines = []

    av = av.copy()
    av["datetime"] = pd.to_datetime(av["datetime"], utc=True)

    n_hours = av["datetime"].nunique()
    n_pairs = av[["node_id", "technology"]].drop_duplicates().shape[0]
    expected_rows = n_hours * n_pairs

    # Row count
    lines.append("=== Node availability checks ===")
    lines.append(f"Rows: {len(av):,} | Hours: {n_hours:,} | Node-tech pairs: {n_pairs:,} | Expected: {expected_rows:,}")
    lines.append("Row count: perfect" if len(av) == expected_rows else "Row count: FAILED")

    # Coverage: no dups / no NaNs
    dup = av.duplicated(subset=["datetime", "node_id", "technology"]).any()
    nan = av["available_mwh"].isna().any()
    lines.append("Coverage: complete" if (not dup and not nan) else "Coverage: FAILED (duplicates or NaNs)")

    # Physics: non-negative
    neg = (av["available_mwh"] < -1e-9).any()
    lines.append("Physics: credible" if not neg else "Physics: FAILED (negative values)")

    # Physics: availability should not exceed capacity * 1h (allow tiny tolerance)
    # Join capacity MW per node-tech
    av2 = av.merge(cap, on=["node_id", "technology"], how="left")
    if av2["p_rated_mw"].isna().any():
        lines.append("WARNING: some availability rows have no matching capacity (check generators.csv node_id/technology labels)")
    else:
        exceed = (av2["available_mwh"] > (av2["p_rated_mw"] * 1.000001)).mean()
        lines.append(f"Capacity bound exceed share: {exceed:.3%} (should be ~0%)")

    # Magnitude: quick stats by tech
    bytech = av.groupby("technology")["available_mwh"].agg(["mean", "max", "sum"])
    for tech, row in bytech.iterrows():
        lines.append(
            f"{tech.title()} | mean={row['mean']:.3f} MWh | max={row['max']:.3f} MWh | total={row['sum'] / 1e3:.2f} GWh"
        )

    return lines


def _basic_checks_region(av_reg: pd.DataFrame) -> list[str]:
    lines = []
    av_reg = av_reg.copy()
    av_reg["datetime"] = pd.to_datetime(av_reg["datetime"], utc=True)

    n_hours = av_reg["datetime"].nunique()
    n_pairs = av_reg[["region", "technology"]].drop_duplicates().shape[0]
    expected_rows = n_hours * n_pairs

    lines.append("\n=== Region availability checks ===")
    lines.append(f"Rows: {len(av_reg):,} | Hours: {n_hours:,} | Region-tech pairs: {n_pairs:,} | Expected: {expected_rows:,}")
    lines.append("Row count: perfect" if len(av_reg) == expected_rows else "Row count: FAILED")

    dup = av_reg.duplicated(subset=["datetime", "region", "technology"]).any()
    nan = av_reg["available_mwh"].isna().any()
    lines.append("Coverage: complete" if (not dup and not nan) else "Coverage: FAILED (duplicates or NaNs)")

    neg = (av_reg["available_mwh"] < -1e-9).any()
    lines.append("Physics: credible" if not neg else "Physics: FAILED (negative values)")

    bytech = av_reg.groupby("technology")["available_mwh"].agg(["mean", "max", "sum"])
    for tech, row in bytech.iterrows():
        lines.append(
            f"{tech.title()} | mean={row['mean']:.3f} MWh | max={row['max']:.3f} MWh | total={row['sum'] / 1e3:.2f} GWh"
        )

    return lines


def _top_regions(av_reg: pd.DataFrame, tech: str, k: int = 3) -> pd.DataFrame:
    t = av_reg[av_reg["technology"] == tech].copy()
    out = (
        t.groupby("region", as_index=False)["available_mwh"]
        .sum()
        .sort_values("available_mwh", ascending=False)
        .head(k)
    )
    out["total_GWh"] = out["available_mwh"] / 1e3
    return out[["region", "total_GWh"]]


def _top_nodes(av_node: pd.DataFrame, nodes: pd.DataFrame, tech: str, k: int = 5) -> pd.DataFrame:
    t = av_node[av_node["technology"] == tech].copy()
    out = (
        t.groupby("node_id", as_index=False)["available_mwh"]
        .sum()
        .sort_values("available_mwh", ascending=False)
        .head(k)
    )
    out = out.merge(nodes[["node_id", "region"]], on="node_id", how="left")
    out["total_GWh"] = out["available_mwh"] / 1e3
    return out[["node_id", "region", "total_GWh"]]


def main(year: int) -> None:
    node_path = Path(f"data/processed/availability_node_hourly/availability_nodes_{year}.parquet")
    reg_path = Path(f"data/processed/availability_region_hourly/availability_regions_{year}.parquet")
    if not node_path.exists():
        raise FileNotFoundError(f"Missing: {node_path} (run build_availability first)")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing: {reg_path} (run build_availability first)")

    av_node = pd.read_parquet(node_path)
    av_reg = pd.read_parquet(reg_path)

    # Load supporting metadata
    gens = pd.read_csv("data/raw/generators.csv")
    gens["node_id"] = gens["node_id"].astype(str)
    gens["technology"] = gens["technology"].astype(str).str.strip().str.lower()
    cap = (
        gens.groupby(["node_id", "technology"], as_index=False)
        .agg(p_rated_mw=("rated_MW", "sum"))
    )

    nodes = pd.read_csv("data/raw/nodes.csv").rename(columns={"Node": "node_id", "Region": "region"})
    nodes["node_id"] = nodes["node_id"].astype(str)

    print(f"=== Availability checker: {year} ===")
    print(f"Node file:   {node_path}")
    print(f"Region file: {reg_path}")

    # Checks
    for line in _basic_checks_node(av_node, cap):
        print(line)
    for line in _basic_checks_region(av_reg):
        print(line)

    # Rankings
    print("\n=== Top 3 regions by total production (GWh) ===")
    for tech in ["wind", "solar"]:
        top = _top_regions(av_reg, tech, k=3)
        print(f"\n{tech.title()}:")
        if len(top) == 0:
            print("  (no data)")
        else:
            print(top.to_string(index=False, formatters={"total_GWh": lambda x: f"{x:.2f}"}))

    print("\n=== Top 5 nodes by total production (GWh) ===")
    for tech in ["wind", "solar"]:
        topn = _top_nodes(av_node, nodes, tech, k=5)
        print(f"\n{tech.title()}:")
        if len(topn) == 0:
            print("  (no data)")
        else:
            print(topn.to_string(index=False, formatters={"total_GWh": lambda x: f"{x:.2f}"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity checks + rankings for node and region availability outputs.")
    parser.add_argument("--year", type=int, required=True, help="Year to check (e.g. 2024)")
    args = parser.parse_args()
    main(args.year)
