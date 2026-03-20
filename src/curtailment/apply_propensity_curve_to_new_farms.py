from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


CURVE_CSV = Path("data/processed/exports/curtailment/curtailment_propensity_curve_bins_p3.csv")
AVAIL_XLSX = Path("data/processed/exports/hourly_availability_by_node_phase3.xlsx")
OUT_XLSX = Path("data/processed/exports/curtailment/new_farms_hourly_constraints_phase3.xlsx")
AVAIL_SHEET: str | int | None = "long"

# EDIT THESE (node_id strings must match the Excel)
RATED_MW_BY_NODE = {
    "Phase2Cork": 450.0,
    "Phase2Wexford": 450.0,
    "BallinaP3":500,
    "GalwayP3":500,
    "MoneypointP3":900,

}

BIN_WIDTH_PCT = 10
INCLUDE_OVER_100 = False


def build_bins(bin_width_pct: int, include_over_100: bool) -> tuple[list[float], list[str]]:
    bw = bin_width_pct / 100.0
    edges = np.round(np.arange(0, 1.0 + bw, bw), 10).tolist()
    if edges[-1] < 1.0:
        edges.append(1.0)
    labels = [f"{int(100*edges[i]):02d}-{int(100*edges[i+1]):02d}%" for i in range(len(edges) - 1)]
    if include_over_100:
        edges = edges + [np.inf]
        labels = labels + [">100%"]
    return edges, labels


def coerce_datetime(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        try:
            return s.dt.tz_convert(None)
        except Exception:
            return s
    try:
        dt = pd.to_datetime(s, unit="ms", utc=True)
        return dt.dt.tz_convert(None)
    except Exception:
        return pd.to_datetime(s, errors="coerce")


def _norm_col_name(value: object) -> str:
    s = str(value).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _find_header_row(preview: pd.DataFrame, aliases_by_target: dict[str, set[str]]) -> tuple[int | None, dict[str, str]]:
    preview_str = preview.astype(str).apply(lambda col: col.map(_norm_col_name))

    header_row = None
    rename_map: dict[str, str] = {}
    required_targets = {"datetime", "node_id", "technology", "available_mwh", "year"}

    for i in range(len(preview_str)):
        row_vals = set(preview_str.iloc[i].dropna().tolist())
        row_map: dict[str, str] = {}
        for target, aliases in aliases_by_target.items():
            for alias in aliases:
                if alias in row_vals:
                    row_map[alias] = target
                    break
        if len(set(row_map.values()).intersection(required_targets)) >= 4:
            header_row = i
            rename_map = row_map
            break
    return header_row, rename_map


def _read_sheet_with_detected_header(path: Path, sheet_name: str | int) -> pd.DataFrame | None:
    """
    Reads an Excel sheet where the header row might not be row 0 (e.g., merged cells / blank rows).
    It scans the first 50 rows for a row containing expected header labels (incl. common aliases)
    and re-reads with that row as header.
    """
    preview = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=50)
    aliases_by_target = {
        "datetime": {"datetime", "date_time", "timestamp", "time", "date"},
        "year": {"year"},
        "node_id": {"node_id", "node", "nodeid"},
        "technology": {"technology", "tech"},
        "available_mwh": {"available_mwh", "available", "avail_mwh", "availability_mwh", "available_mw"},
    }
    header_row, alias_to_target = _find_header_row(preview, aliases_by_target)

    if header_row is None:
        return None

    df = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
    norm_cols = [_norm_col_name(c) for c in df.columns]
    df.columns = [alias_to_target.get(c, c) for c in norm_cols]
    # drop completely empty columns
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]
    return df


def read_availability_xlsx_with_header_detection(path: Path, sheet_name: str | int | None = "long") -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    candidate_sheets: list[str | int] = []

    if sheet_name is not None:
        candidate_sheets.append(sheet_name)
    candidate_sheets += [s for s in xls.sheet_names if s not in candidate_sheets]

    req = {"datetime", "year", "node_id", "technology", "available_mwh"}
    for s in candidate_sheets:
        try:
            df = _read_sheet_with_detected_header(path, sheet_name=s)
        except Exception:
            continue
        if df is not None and req.issubset(set(df.columns)):
            return df

    raise ValueError(
        "Could not detect a suitable sheet/header containing expected columns "
        f"{sorted(req)}. Checked sheets: {xls.sheet_names}. "
        "Use the 'long' sheet, or ensure one sheet has headers like: datetime, year, node_id, technology, available_mwh."
    )


def main() -> None:
    # ---- load curve ----
    curve = pd.read_csv(CURVE_CSV)
    if not {"bin", "y_avail_final"}.issubset(curve.columns):
        raise KeyError(f"Curve CSV must contain columns ['bin','y_avail_final']. Found: {list(curve.columns)}")
    curve = curve[["bin", "y_avail_final"]].copy()

    # ---- load availability (robust header detection) ----
    df = read_availability_xlsx_with_header_detection(AVAIL_XLSX, sheet_name=AVAIL_SHEET)

    # normalise column names
    df.columns = [str(c).strip().lower() for c in df.columns]

    # accept slight variants
    rename_map = {}
    if "node" in df.columns and "node_id" not in df.columns:
        rename_map["node"] = "node_id"
    if "available_mwh" not in df.columns:
        for alt in ["available", "avail_mwh", "availability_mwh", "available_mw", "available"]:
            if alt in df.columns:
                rename_map[alt] = "available_mwh"
                break
    df = df.rename(columns=rename_map)

    req = {"datetime", "year", "node_id", "technology", "available_mwh"}
    miss = req - set(df.columns)
    if miss:
        raise KeyError(f"Availability XLSX missing columns {miss}. Found: {list(df.columns)}")

    df = df.copy()
    df["datetime"] = coerce_datetime(df["datetime"])
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()
    df["available_mwh"] = pd.to_numeric(df["available_mwh"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["datetime", "node_id", "available_mwh"])

    # wind only
    df = df[df["technology"] == "wind"].copy()

    # ---- attach rated MW ----
    if not RATED_MW_BY_NODE:
        nodes = sorted(df["node_id"].unique().tolist())
        raise ValueError(
            "RATED_MW_BY_NODE is empty. Add your two node_id keys with rated MW.\n"
            f"node_id values found: {nodes}"
        )

    rated_map = pd.Series(RATED_MW_BY_NODE, name="rated_mw").rename_axis("node_id").reset_index()
    df = df.merge(rated_map, on="node_id", how="left")

    missing_nodes = sorted(df.loc[df["rated_mw"].isna(), "node_id"].unique().tolist())
    if missing_nodes:
        raise ValueError(f"Missing rated MW for node_id(s): {missing_nodes}. Add to RATED_MW_BY_NODE.")

    # ---- compute x + bins ----
    df["x"] = df["available_mwh"] / df["rated_mw"]
    df["x"] = df["x"].replace([np.inf, -np.inf], np.nan)

    edges, labels = build_bins(BIN_WIDTH_PCT, INCLUDE_OVER_100)
    df["bin"] = pd.cut(df["x"], bins=edges, labels=labels, include_lowest=True, right=True)

    # ---- apply curve ----
    df = df.merge(curve, on="bin", how="left")

    if df["y_avail_final"].isna().any():
        bad = df[df["y_avail_final"].isna()][["node_id", "datetime", "x", "bin"]].head(20)
        raise ValueError(
            "Some rows did not match a curve bin (y_avail_final is NaN). "
            "Check BIN_WIDTH_PCT/INCLUDE_OVER_100 matches the curve build.\n"
            f"Example bad rows:\n{bad}"
        )

    df["curtailment_hat_mwh"] = (df["available_mwh"] * df["y_avail_final"]).clip(lower=0)
    df["curtailment_hat_mwh"] = np.minimum(df["curtailment_hat_mwh"], df["available_mwh"])
    df["net_gen_hat_mwh"] = (df["available_mwh"] - df["curtailment_hat_mwh"]).clip(lower=0)

    df["curtailment_frac_of_avail"] = df["curtailment_hat_mwh"] / (df["available_mwh"] + 1e-12)
    df["curtailment_frac_of_rated"] = df["curtailment_hat_mwh"] / df["rated_mw"]

    # ---- summaries ----
    summary = (
        df.groupby(["node_id", "year"], as_index=False)
        .agg(
            rated_mw=("rated_mw", "first"),
            avail_mwh=("available_mwh", "sum"),
            curtailed_mwh=("curtailment_hat_mwh", "sum"),
            net_gen_mwh=("net_gen_hat_mwh", "sum"),
            mean_x=("x", "mean"),
            p95_x=("x", lambda s: float(np.nanquantile(s.to_numpy(), 0.95))),
        )
    )
    summary["curtailment_share_of_avail"] = summary["curtailed_mwh"] / (summary["avail_mwh"] + 1e-12)

    by_bin = (
        df.groupby(["node_id", "bin"], as_index=False)
        .agg(
            hours=("datetime", "count"),
            avail_mwh=("available_mwh", "sum"),
            curtailed_mwh=("curtailment_hat_mwh", "sum"),
            mean_x=("x", "mean"),
        )
    )
    by_bin["curtailment_share_of_avail"] = by_bin["curtailed_mwh"] / (by_bin["avail_mwh"] + 1e-12)

    # ---- export ----
    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        df.sort_values(["node_id", "datetime"]).to_excel(writer, sheet_name="hourly", index=False)
        summary.to_excel(writer, sheet_name="summary", index=False)
        by_bin.to_excel(writer, sheet_name="by_bin", index=False)

    print("[ok] Wrote:", OUT_XLSX)
    print("[info] Rows:", len(df), "Nodes:", df["node_id"].nunique())
    print("[info] node_id found:", sorted(df["node_id"].unique().tolist()))


if __name__ == "__main__":
    main()
