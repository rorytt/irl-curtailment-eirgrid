from __future__ import annotations

from pathlib import Path
import math
import pandas as pd

EXCEL_MAX_ROWS = 1_048_576  # hard Excel limit per sheet


def _write_df_split_sheets(
    df: pd.DataFrame,
    writer: pd.ExcelWriter,
    base_sheet: str,
    max_rows: int = EXCEL_MAX_ROWS,
) -> None:
    """
    Write df to one or more sheets. If df exceeds max_rows-1 (header row), split into chunks.
    """
    if df.empty:
        df.to_excel(writer, sheet_name=base_sheet[:31], index=False)
        return

    chunk_size = max_rows - 1  # reserve 1 row for headers
    n = len(df)
    n_chunks = math.ceil(n / chunk_size)

    if n_chunks == 1:
        df.to_excel(writer, sheet_name=base_sheet[:31], index=False)
        return

    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, n)
        sheet = f"{base_sheet}_{i+1}"
        df.iloc[start:end].to_excel(writer, sheet_name=sheet[:31], index=False)


def export_parquets_to_excel(
    in_dir: Path,
    out_dir: Path,
    years: list[int] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    p_meta = in_dir / "wind_curtailment_meta_month.parquet"
    p_node = in_dir / "wind_curtailment_node_hour.parquet"
    p_reg = in_dir / "wind_curtailment_region_hour.parquet"

    if not p_meta.exists() or not p_node.exists() or not p_reg.exists():
        raise FileNotFoundError(
            "Missing one or more parquet outputs in "
            f"{in_dir}. Expected:\n"
            f"- {p_meta}\n- {p_node}\n- {p_reg}"
        )

    df_meta = pd.read_parquet(p_meta)
    df_node = pd.read_parquet(p_node)
    df_reg = pd.read_parquet(p_reg)

    # Optional year filter
    if years is not None:
        if "weather_year" in df_meta.columns:
            df_meta = df_meta[df_meta["weather_year"].isin(years)]
        if "weather_year" in df_node.columns:
            df_node = df_node[df_node["weather_year"].isin(years)]
        if "weather_year" in df_reg.columns:
            df_reg = df_reg[df_reg["weather_year"].isin(years)]

    # Sort for readability
    if {"weather_year", "month"}.issubset(df_meta.columns):
        df_meta = df_meta.sort_values(["weather_year", "month"])
    if {"weather_year", "datetime", "region"}.issubset(df_reg.columns):
        df_reg = df_reg.sort_values(["weather_year", "datetime", "region"])
    if {"weather_year", "datetime", "region", "node_id"}.issubset(df_node.columns):
        df_node = df_node.sort_values(["weather_year", "datetime", "region", "node_id"])

    # ---- 1) meta_month ----
    out_meta = out_dir / "wind_curtailment_meta_month.xlsx"
    with pd.ExcelWriter(out_meta, engine="openpyxl") as writer:
        _write_df_split_sheets(df_meta, writer, base_sheet="meta_month")
    print(f"Saved: {out_meta}")

    # ---- 2) region_hour (split by year) ----
    out_reg = out_dir / "wind_curtailment_region_hour.xlsx"
    with pd.ExcelWriter(out_reg, engine="openpyxl") as writer:
        if "weather_year" in df_reg.columns:
            for y, d in df_reg.groupby("weather_year", sort=True):
                _write_df_split_sheets(d, writer, base_sheet=f"y{int(y)}")
        else:
            _write_df_split_sheets(df_reg, writer, base_sheet="region_hour")
    print(f"Saved: {out_reg}")

    # ---- 3) node_hour (split by year + chunk if needed) ----
    out_node = out_dir / "wind_curtailment_node_hour.xlsx"
    with pd.ExcelWriter(out_node, engine="openpyxl") as writer:
        if "weather_year" in df_node.columns:
            for y, d in df_node.groupby("weather_year", sort=True):
                _write_df_split_sheets(d, writer, base_sheet=f"y{int(y)}")
        else:
            _write_df_split_sheets(df_node, writer, base_sheet="node_hour")
    print(f"Saved: {out_node}")


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in_dir",
        default="data/processed/curtailment_counterfactual_wind",
        help="Where the parquet outputs live",
    )
    ap.add_argument(
        "--out_dir",
        default="data/processed/exports/curtailment",
        help="Where to write the Excel files",
    )
    ap.add_argument(
        "--years",
        default="",
        help="Optional comma-separated years to export (e.g. 2025 or 2021,2024,2025)",
    )
    args = ap.parse_args()

    years = [int(x) for x in args.years.split(",") if x.strip()] or None

    export_parquets_to_excel(
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        years=years,
    )


if __name__ == "__main__":
    main()
