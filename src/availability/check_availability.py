# src/availability/check_availability.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def _basic_checks_node(av: pd.DataFrame, cap: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    av = av.copy()
    av["datetime"] = pd.to_datetime(av["datetime"], utc=True)
    av["node_id"] = av["node_id"].astype(str)
    av["technology"] = av["technology"].astype(str).str.strip().str.lower()

    n_hours = av["datetime"].nunique()
    n_pairs = av[["node_id", "technology"]].drop_duplicates().shape[0]
    expected_rows = n_hours * n_pairs

    lines.append("=== Node availability checks ===")
    lines.append(f"Rows: {len(av):,} | Hours: {n_hours:,} | Node-tech pairs: {n_pairs:,} | Expected: {expected_rows:,}")
    lines.append("Row count: perfect" if len(av) == expected_rows else "Row count: FAILED")

    dup = av.duplicated(subset=["datetime", "node_id", "technology"]).any()
    nan = av["available_mwh"].isna().any()
    lines.append("Coverage: complete" if (not dup and not nan) else "Coverage: FAILED (duplicates or NaNs)")

    neg = (av["available_mwh"] < -1e-9).any()
    lines.append("Physics: credible" if not neg else "Physics: FAILED (negative values)")

    # capacity bound check: availability_mwh (1h) should be <= installed MW (allow tiny tol)
    cap2 = cap.copy()
    cap2["node_id"] = cap2["node_id"].astype(str)
    cap2["technology"] = cap2["technology"].astype(str).str.strip().str.lower()

    av2 = av.merge(cap2, on=["node_id", "technology"], how="left")

    if av2["p_rated_mw"].isna().any():
        n_missing = int(av2["p_rated_mw"].isna().sum())
        lines.append(f"WARNING: {n_missing:,} rows have no matching capacity (check generators.csv labels / node_id types)")
    else:
        exceed_share = (av2["available_mwh"] > (av2["p_rated_mw"] * 1.000001)).mean()
        lines.append(f"Capacity bound exceed share: {exceed_share:.3%} (should be ~0%)")

    # quick magnitudes by tech
    bytech = av.groupby("technology")["available_mwh"].agg(["mean", "max", "sum"])
    for tech, row in bytech.iterrows():
        lines.append(
            f"{tech.title()} | mean={row['mean']:.3f} MWh | max={row['max']:.3f} MWh | total={row['sum']/1e3:.2f} GWh"
        )

    pairs_by_tech = av[["node_id", "technology"]].drop_duplicates().groupby("technology").size()
    for tech, cnt in pairs_by_tech.items():
        lines.append(f"{tech.title()} node-tech pairs: {int(cnt)}")

    return lines


def _basic_checks_region(av_reg: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    df = av_reg.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()

    n_hours = df["datetime"].nunique()
    n_pairs = df[["region", "technology"]].drop_duplicates().shape[0]
    expected_rows = n_hours * n_pairs

    lines.append("=== Region availability checks ===")
    lines.append(f"Rows: {len(df):,} | Hours: {n_hours:,} | Region-tech pairs: {n_pairs:,} | Expected: {expected_rows:,}")
    lines.append("Row count: perfect" if len(df) == expected_rows else "Row count: FAILED")

    dup = df.duplicated(subset=["datetime", "region", "technology"]).any()
    nan = df["available_mwh"].isna().any()
    lines.append("Coverage: complete" if (not dup and not nan) else "Coverage: FAILED (duplicates or NaNs)")

    neg = (df["available_mwh"] < -1e-9).any()
    lines.append("Physics: credible" if not neg else "Physics: FAILED (negative values)")

    bytech = df.groupby("technology")["available_mwh"].agg(["mean", "max", "sum"])
    for tech, row in bytech.iterrows():
        lines.append(
            f"{tech.title()} | mean={row['mean']:.3f} MWh | max={row['max']:.3f} MWh | total={row['sum']/1e3:.2f} GWh"
        )

    return lines


def _top_regions(av_reg: pd.DataFrame, tech: str, k: int = 3) -> pd.DataFrame:
    t = av_reg.copy()
    t["technology"] = t["technology"].astype(str).str.strip().str.lower()
    t = t[t["technology"] == tech]
    out = (
        t.groupby("region", as_index=False)["available_mwh"]
        .sum()
        .sort_values("available_mwh", ascending=False)
        .head(k)
        .copy()
    )
    out["total_GWh"] = out["available_mwh"] / 1e3
    return out[["region", "total_GWh"]]


def _top_nodes(av_node: pd.DataFrame, nodes: pd.DataFrame, tech: str, k: int = 5) -> pd.DataFrame:
    t = av_node.copy()
    t["technology"] = t["technology"].astype(str).str.strip().str.lower()
    t = t[t["technology"] == tech]
    out = (
        t.groupby("node_id", as_index=False)["available_mwh"]
        .sum()
        .sort_values("available_mwh", ascending=False)
        .head(k)
        .copy()
    )
    out = out.merge(nodes[["node_id", "region"]], on="node_id", how="left")
    out["total_GWh"] = out["available_mwh"] / 1e3
    return out[["node_id", "region", "total_GWh"]]


def _system_monthly_table(av_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Monthly system summary (sum all regions each hour), per technology:
      year_month, technology, hours, mean_MW, max_MW, total_GWh
    Adds TOTAL row per technology.
    """
    df = av_reg.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()

    # system total per hour (sum all regions)
    sys_hour = (
        df.groupby(["datetime", "technology"], as_index=False)
        .agg(available_mwh=("available_mwh", "sum"))
    )
    sys_hour["year_month"] = sys_hour["datetime"].dt.to_period("M").astype(str)

    monthly = (
        sys_hour.groupby(["year_month", "technology"], as_index=False)
        .agg(
            hours=("available_mwh", "size"),
            mean_MW=("available_mwh", "mean"),          # MWh per 1h => MW average
            max_MW=("available_mwh", "max"),            # max hourly MW
            total_GWh=("available_mwh", lambda s: s.sum() / 1e3),
        )
        .sort_values(["technology", "year_month"])
        .reset_index(drop=True)
    )

    totals = (
        sys_hour.groupby("technology", as_index=False)
        .agg(
            hours=("available_mwh", "size"),
            mean_MW=("available_mwh", "mean"),
            max_MW=("available_mwh", "max"),
            total_GWh=("available_mwh", lambda s: s.sum() / 1e3),
        )
    )
    totals.insert(0, "year_month", "TOTAL")

    return pd.concat([monthly, totals], ignore_index=True)


def _df_to_markdown_codeblock(df: pd.DataFrame) -> str:
    return "```\n" + df.to_string(index=False) + "\n```"


def _write_report(
    year: int,
    node_path: Path,
    reg_path: Path,
    node_check_lines: list[str],
    reg_check_lines: list[str],
    top_region_text: str,
    top_node_text: str,
    monthly_table: pd.DataFrame,
) -> Path:
    out_dir = Path("data/processed/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"availability_report_{year}.md"

    mt = monthly_table.copy()
    mt["mean_MW"] = mt["mean_MW"].map(lambda x: f"{x:.2f}")
    mt["max_MW"] = mt["max_MW"].map(lambda x: f"{x:.2f}")
    mt["total_GWh"] = mt["total_GWh"].map(lambda x: f"{x:.2f}")

    content: list[str] = []
    content.append(f"# Availability Report — {year}\n")
    content.append(f"- Node file: `{node_path}`")
    content.append(f"- Region file: `{reg_path}`\n")

    content.append("## Node availability checks\n")
    content.append("```\n" + "\n".join(node_check_lines) + "\n```\n")

    content.append("## Region availability checks\n")
    content.append("```\n" + "\n".join(reg_check_lines) + "\n```\n")

    content.append("## Top regions\n")
    content.append(top_region_text + "\n")

    content.append("## Top nodes\n")
    content.append(top_node_text + "\n")

    content.append("## Monthly system availability summary\n")
    content.append(
        "Definitions (hourly availability):\n"
        "- mean (MW): average of hourly available power in the month\n"
        "- max (MW): maximum hourly available power in the month\n"
        "- total (GWh): sum of hourly availability (MWh) / 1000\n"
    )
    content.append(_df_to_markdown_codeblock(mt) + "\n")

    out_path.write_text("\n".join(content), encoding="utf-8")
    return out_path


def main(year: int) -> None:
    node_path = Path(f"data/processed/availability_node_hourly/availability_nodes_{year}.parquet")
    reg_path = Path(f"data/processed/availability_region_hourly/availability_regions_{year}.parquet")
    if not node_path.exists():
        raise FileNotFoundError(f"Missing: {node_path} (run build_availability first)")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing: {reg_path} (run build_availability first)")

    av_node = pd.read_parquet(node_path)
    av_reg = pd.read_parquet(reg_path)

    # capacities (MW) per node-tech
    gens = pd.read_csv("data/raw/generators.csv")
    gens["node_id"] = gens["node_id"].astype(str)
    gens["technology"] = gens["technology"].astype(str).str.strip().str.lower()
    cap = (
        gens.groupby(["node_id", "technology"], as_index=False)
        .agg(p_rated_mw=("rated_MW", "sum"))
    )

    # node -> region
    nodes = pd.read_csv("data/raw/nodes.csv").rename(columns={"Node": "node_id", "Region": "region"})
    nodes["node_id"] = nodes["node_id"].astype(str)

    print(f"=== Availability checker: {year} ===")
    print(f"Node file:   {node_path}")
    print(f"Region file: {reg_path}")

    # checks (store + print)
    node_check_lines = _basic_checks_node(av_node, cap)
    reg_check_lines = _basic_checks_region(av_reg)

    for line in node_check_lines:
        print(line)
    print()
    for line in reg_check_lines:
        print(line)

    # top regions text
    top_region_lines: list[str] = []
    top_region_lines.append("=== Top 3 regions by total production (GWh) ===")
    for tech in ["wind", "solar"]:
        top = _top_regions(av_reg, tech, k=3)
        top_region_lines.append(f"\n{tech.title()}:")
        if len(top) == 0:
            top_region_lines.append("  (no data)")
        else:
            top_region_lines.append(top.to_string(index=False, formatters={"total_GWh": lambda x: f"{x:.2f}"}))
    top_region_text = "```\n" + "\n".join(top_region_lines) + "\n```"
    print("\n" + "\n".join(top_region_lines))

    # top nodes text
    top_node_lines: list[str] = []
    top_node_lines.append("=== Top 5 nodes by total production (GWh) ===")
    for tech in ["wind", "solar"]:
        topn = _top_nodes(av_node, nodes, tech, k=5)
        top_node_lines.append(f"\n{tech.title()}:")
        if len(topn) == 0:
            top_node_lines.append("  (no data)")
        else:
            top_node_lines.append(topn.to_string(index=False, formatters={"total_GWh": lambda x: f"{x:.2f}"}))
    top_node_text = "```\n" + "\n".join(top_node_lines) + "\n```"
    print("\n" + "\n".join(top_node_lines))

    # monthly table + report
    monthly_table = _system_monthly_table(av_reg)
    report_path = _write_report(
        year=year,
        node_path=node_path,
        reg_path=reg_path,
        node_check_lines=node_check_lines,
        reg_check_lines=reg_check_lines,
        top_region_text=top_region_text,
        top_node_text=top_node_text,
        monthly_table=monthly_table,
    )
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanity checks + rankings + report for node & region availability."
    )
    parser.add_argument("--year", type=int, required=True, help="Year to check (e.g. 2024)")
    args = parser.parse_args()
    main(args.year)
