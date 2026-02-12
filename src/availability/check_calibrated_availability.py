# src/availability/check_calibrated_availability.py
from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


def _basic_checks_node(av: pd.DataFrame, cap: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    df = av.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["node_id"] = df["node_id"].astype(str)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()

    n_hours = df["datetime"].nunique()
    n_pairs = df[["node_id", "technology"]].drop_duplicates().shape[0]
    expected_rows = n_hours * n_pairs

    lines.append("=== Node availability checks (CALIBRATED) ===")
    lines.append(f"Rows: {len(df):,} | Hours: {n_hours:,} | Node-tech pairs: {n_pairs:,} | Expected: {expected_rows:,}")
    lines.append("Row count: perfect" if len(df) == expected_rows else "Row count: FAILED")

    dup = df.duplicated(subset=["datetime", "node_id", "technology"]).any()
    nan = df["available_mwh"].isna().any()
    lines.append("Coverage: complete" if (not dup and not nan) else "Coverage: FAILED (duplicates or NaNs)")

    neg = (df["available_mwh"] < -1e-9).any()
    lines.append("Physics: credible" if not neg else "Physics: FAILED (negative values)")

    # capacity bound check (post-calibration this can exceed installed MW if factors > 1)
    cap2 = cap.copy()
    cap2["node_id"] = cap2["node_id"].astype(str)
    cap2["technology"] = cap2["technology"].astype(str).str.strip().str.lower()

    df2 = df.merge(cap2, on=["node_id", "technology"], how="left")
    if df2["p_rated_mw"].isna().any():
        n_missing = int(df2["p_rated_mw"].isna().sum())
        lines.append(f"WARNING: {n_missing:,} rows have no matching capacity (check generators.csv labels / node_id types)")
    else:
        exceed_share = (df2["available_mwh"] > (df2["p_rated_mw"] * 1.000001)).mean()
        lines.append(f"Capacity bound exceed share: {exceed_share:.3%} (can be >0 after calibration)")

    bytech = df.groupby("technology")["available_mwh"].agg(["mean", "max", "sum"])
    for tech, row in bytech.iterrows():
        lines.append(
            f"{tech.title()} | mean={row['mean']:.3f} MWh | max={row['max']:.3f} MWh | total={row['sum']/1e3:.2f} GWh"
        )

    return lines


def _basic_checks_region(av_reg: pd.DataFrame) -> list[str]:
    lines: list[str] = []

    df = av_reg.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()

    n_hours = df["datetime"].nunique()
    n_pairs = df[["region", "technology"]].drop_duplicates().shape[0]
    expected_rows = n_hours * n_pairs

    lines.append("=== Region availability checks (CALIBRATED) ===")
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
    df = av_reg.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()

    sys_hour = (
        df.groupby(["datetime", "technology"], as_index=False)
        .agg(available_mwh=("available_mwh", "sum"))
    )
    sys_hour["year_month"] = sys_hour["datetime"].dt.to_period("M").astype(str)

    monthly = (
        sys_hour.groupby(["year_month", "technology"], as_index=False)
        .agg(
            hours=("available_mwh", "size"),
            mean_MW=("available_mwh", "mean"),
            max_MW=("available_mwh", "max"),
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
    node_head: pd.DataFrame,
    reg_head: pd.DataFrame,
    node_check_lines: list[str],
    reg_check_lines: list[str],
    top_region_text: str,
    top_node_text: str,
    monthly_table: pd.DataFrame,
) -> Path:
    out_dir = Path("data/processed/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"availability_report_{year}_calibrated.md"

    mt = monthly_table.copy()
    mt["mean_MW"] = mt["mean_MW"].map(lambda x: f"{x:.2f}")
    mt["max_MW"] = mt["max_MW"].map(lambda x: f"{x:.2f}")
    mt["total_GWh"] = mt["total_GWh"].map(lambda x: f"{x:.2f}")

    content: list[str] = []
    content.append(f"# Calibrated Availability Report — {year}\n")
    content.append(f"- Node file: `{node_path}`")
    content.append(f"- Region file: `{reg_path}`\n")

    content.append("## Parquet preview\n")
    content.append("### Node parquet (head)\n")
    content.append(_df_to_markdown_codeblock(node_head) + "\n")
    content.append("### Region parquet (head)\n")
    content.append(_df_to_markdown_codeblock(reg_head) + "\n")

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
        "\nNote: values are *calibrated* using monthly factors; this can cause some hours to exceed installed MW if a factor > 1.\n"
    )
    content.append(_df_to_markdown_codeblock(mt) + "\n")

    out_path.write_text("\n".join(content), encoding="utf-8")
    return out_path


def main(year: int) -> None:
    node_path = Path(f"data/processed/availability_node_hourly_calibrated/availability_nodes_{year}_calibrated.parquet")
    reg_path = Path(f"data/processed/availability_region_hourly_calibrated/availability_regions_{year}_calibrated.parquet")

    if not node_path.exists():
        raise FileNotFoundError(f"Missing calibrated node availability: {node_path} (run calibrate_monthly)")
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing calibrated region availability: {reg_path} (run calibrate_monthly)")

    av_node = pd.read_parquet(node_path)
    av_reg = pd.read_parquet(reg_path)

    # Parquet head preview (print + include in report)
    node_head = av_node.head(10).copy()
    reg_head = av_reg.head(10).copy()

    print(f"=== Calibrated availability checker: {year} ===")
    print(f"Node file:   {node_path}")
    print(f"Region file: {reg_path}")
    print("\n--- Node parquet head (10 rows) ---")
    print(node_head.to_string(index=False))
    print("\n--- Region parquet head (10 rows) ---")
    print(reg_head.to_string(index=False))

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

    node_check_lines = _basic_checks_node(av_node, cap)
    reg_check_lines = _basic_checks_region(av_reg)

    print("\n" + "\n".join(node_check_lines))
    print("\n" + "\n".join(reg_check_lines))

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

    monthly_table = _system_monthly_table(av_reg)

    report_path = _write_report(
        year=year,
        node_path=node_path,
        reg_path=reg_path,
        node_head=node_head,
        reg_head=reg_head,
        node_check_lines=node_check_lines,
        reg_check_lines=reg_check_lines,
        top_region_text=top_region_text,
        top_node_text=top_node_text,
        monthly_table=monthly_table,
    )
    print(f"\nWrote report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a QA report for CALIBRATED availability outputs.")
    parser.add_argument("--year", type=int, required=True, help="Year to check (e.g. 2024, 2025)")
    args = parser.parse_args()
    main(args.year)
