# src/curtailment/curtailment_share_by_availability_bins.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# CONFIG
# =========================
GENERATOR_CSV = Path("data/raw/original/generators.csv")

AVAILABILITY_PARQUET = Path(
    "data/processed/original/availability_node_hourly_calibrated/availability_nodes_2025_calibrated.parquet"
)

CURTAILMENT_NODE_HOUR_PARQUET = Path(
    "data/processed/curtailment_counterfactual_wind/wind_curtailment_node_hour_y2025.parquet"
)

OUT_DIR = Path("data/processed/exports/curtailment")

# SW+SE corridor in your generator.csv uses region codes
LOCAL_REGIONS = {"W", "NW"}

BIN_WIDTH_PCT = 10
INCLUDE_OVER_100 = False

# shrinkage strength (bigger = more conservative / more national)
SHRINK_K = 15000

# Data/time assumptions
TIMESTEP_HOURS = 1.0  # only used if curtailment is in MW rather than MWh

# Safety
EPS_AVAIL = 1e-9
CLIP_AVAILABLE_TO_RATED = True


# =========================
# Helpers
# =========================
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def infer_curtailment_col(df: pd.DataFrame) -> tuple[str, str]:
    """Returns (colname, unit) where unit is 'mwh' or 'mw'."""
    mwh_candidates = [
        # your 2025 parquet
        "c_mwh",
        # common names
        "curtailment_mwh",
        "allocated_mwh",
        "dd_mwh",
        "dispatch_down_mwh",
        "DD_mwh",
        "curtailment",
        "allocated",
        "dd",
    ]
    for c in mwh_candidates:
        if c in df.columns:
            return c, "mwh"

    mw_candidates = [
        "curtailment_mw",
        "allocated_mw",
        "dd_mw",
        "dispatch_down_mw",
        "DD_mw",
    ]
    for c in mw_candidates:
        if c in df.columns:
            return c, "mw"

    raise KeyError(f"Could not infer curtailment column. Found: {list(df.columns)}")


def coerce_datetime(series: pd.Series) -> pd.Series:
    """Returns tz-naive datetime64[ns], handles ms epoch and tz-aware."""
    if pd.api.types.is_datetime64_any_dtype(series):
        try:
            return series.dt.tz_convert(None)
        except Exception:
            return series

    # Try epoch ms
    try:
        dt = pd.to_datetime(series, unit="ms", utc=True)
        return dt.dt.tz_convert(None)
    except Exception:
        dt = pd.to_datetime(series, utc=True, errors="coerce")
        return dt.dt.tz_convert(None)


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


def q(x: pd.Series, p: float) -> float:
    x = x.dropna()
    if len(x) == 0:
        return np.nan
    return float(np.nanquantile(x.to_numpy(), p))


def agg_curve(df: pd.DataFrame, bin_col: str, weight_col: str) -> pd.DataFrame:
    """Curve stats per bin. Includes ratio-of-sums (stable) and distribution stats."""
    def wavg(s: pd.Series) -> float:
        w = df.loc[s.index, weight_col].to_numpy(dtype=float)
        v = s.to_numpy(dtype=float)
        mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if mask.sum() == 0:
            return np.nan
        return float(np.average(v[mask], weights=w[mask]))

    g = df.groupby(bin_col, dropna=False)

    out = g.agg(
        n_rows=("node_id", "size"),
        n_nodes=("node_id", "nunique"),
        sum_curt_mwh=("curtailment_mwh", "sum"),
        sum_avail_mwh=("available_mwh", "sum"),
        sum_rated_mw=(weight_col, "sum"),
        mean_x=("x", "mean"),
        p50_x=("x", "median"),
        mean_y_avail=("y_avail", "mean"),
        p50_y_avail=("y_avail", "median"),
        p10_y_avail=("y_avail", lambda s: q(s, 0.10)),
        p90_y_avail=("y_avail", lambda s: q(s, 0.90)),
        wmean_y_avail=("y_avail", wavg),
    ).reset_index()

    # ratio-of-sums (what we use for the curve)
    out["ros_y_avail"] = out["sum_curt_mwh"] / (out["sum_avail_mwh"] + EPS_AVAIL)
    return out


def shrink(local: pd.Series, nat: pd.Series, n_local: pd.Series, k: float) -> tuple[pd.Series, pd.Series]:
    """y_final = w*local + (1-w)*nat, w = n_local/(n_local+k)."""
    w = n_local / (n_local + k)
    y_final = w * local + (1.0 - w) * nat
    return y_final, w


# =========================
# Main
# =========================
def main() -> None:
    ensure_dir(OUT_DIR)

    # -----------------------------
    # 1) generators: node_id -> rated MW + region (wind only)
    # -----------------------------
    gens = pd.read_csv(GENERATOR_CSV)
    req = {"region", "node_id", "rated_MW", "technology"}
    miss = req - set(gens.columns)
    if miss:
        raise KeyError(f"generators.csv missing {miss}. Found: {list(gens.columns)}")

    gens = gens.copy()
    gens["technology"] = gens["technology"].astype(str).str.strip().str.lower()
    gens = gens[gens["technology"] == "wind"].copy()
    gens["rated_MW"] = pd.to_numeric(gens["rated_MW"], errors="coerce")
    gens = gens.dropna(subset=["node_id", "region", "rated_MW"])

    # normalise region strings (your file uses SW/SE)
    gens["region"] = gens["region"].astype(str).str.strip().str.upper()

    # if node_id appears in multiple regions, pick most common
    region_counts = gens.groupby(["node_id", "region"]).size().reset_index(name="n")
    region_pick = region_counts.sort_values(["node_id", "n"], ascending=[True, False]).drop_duplicates("node_id")
    node_region = region_pick[["node_id", "region"]].copy()

    multi_region_nodes = region_counts["node_id"].value_counts()
    multi_region_nodes = multi_region_nodes[multi_region_nodes > 1]
    if len(multi_region_nodes) > 0:
        print(f"[warn] {len(multi_region_nodes)} node_id(s) appear in multiple regions. Using most common region per node.")

    node_rated = (
        gens.groupby("node_id", as_index=False)["rated_MW"]
        .sum()
        .rename(columns={"rated_MW": "node_rated_mw"})
    )
    node_map = node_rated.merge(node_region, on="node_id", how="left")

    print("[check] generator regions:", sorted(gens["region"].unique().tolist()))
    print("[check] node_map nodes:", node_map["node_id"].nunique())

    # -----------------------------
    # 2) availability (wind only) + attach node_map
    # -----------------------------
    avail = pd.read_parquet(AVAILABILITY_PARQUET).copy()
    req_a = {"datetime", "node_id", "technology", "available_mwh"}
    miss_a = req_a - set(avail.columns)
    if miss_a:
        raise KeyError(f"availability parquet missing {miss_a}. Found: {list(avail.columns)}")

    avail["technology"] = avail["technology"].astype(str).str.strip().str.lower()
    avail = avail[avail["technology"] == "wind"].copy()

    avail["datetime"] = coerce_datetime(avail["datetime"])
    avail["available_mwh"] = pd.to_numeric(avail["available_mwh"], errors="coerce")
    avail = avail.dropna(subset=["datetime", "node_id", "available_mwh"])

    avail = avail.merge(node_map, on="node_id", how="left")
    missing = avail["node_rated_mw"].isna().sum()
    if missing:
        print(f"[warn] Dropping {missing} availability rows: node_id not found in wind generators.csv")
        avail = avail.dropna(subset=["node_rated_mw", "region"])

    avail = avail[avail["node_rated_mw"] > 0].copy()

    if CLIP_AVAILABLE_TO_RATED:
        avail["available_mwh"] = np.minimum(avail["available_mwh"], avail["node_rated_mw"])

    avail["x"] = avail["available_mwh"] / avail["node_rated_mw"]
    avail = avail.replace([np.inf, -np.inf], np.nan).dropna(subset=["x"])

    # -----------------------------
    # 3) curtailment (node-hour)
    # -----------------------------
    curt = pd.read_parquet(CURTAILMENT_NODE_HOUR_PARQUET).copy()

    if "node_id" not in curt.columns:
        for alt in ["node", "Node", "NODE", "nodeid"]:
            if alt in curt.columns:
                curt = curt.rename(columns={alt: "node_id"})
                break
    if "datetime" not in curt.columns:
        for alt in ["Datetime", "time", "timestamp"]:
            if alt in curt.columns:
                curt = curt.rename(columns={alt: "datetime"})
                break
    if "node_id" not in curt.columns or "datetime" not in curt.columns:
        raise KeyError(f"curtailment parquet needs node_id+datetime. Found: {list(curt.columns)}")

    curt_col, curt_unit = infer_curtailment_col(curt)
    curt["datetime"] = coerce_datetime(curt["datetime"])

    curt_val = pd.to_numeric(curt[curt_col], errors="coerce")
    curt["curtailment_mwh"] = curt_val * TIMESTEP_HOURS if curt_unit == "mw" else curt_val
    curt = curt.dropna(subset=["datetime", "node_id", "curtailment_mwh"])
    curt = curt[curt["curtailment_mwh"] > 0].copy()

    # -----------------------------
    # 4) overlap window
    # -----------------------------
    a_min, a_max = avail["datetime"].min(), avail["datetime"].max()
    c_min, c_max = curt["datetime"].min(), curt["datetime"].max()
    start, end = max(a_min, c_min), min(a_max, c_max)
    if start > end:
        raise ValueError(
            f"No overlap in datetimes.\n"
            f"  availability: {a_min} -> {a_max}\n"
            f"  curtailment : {c_min} -> {c_max}\n"
        )
    avail = avail[(avail["datetime"] >= start) & (avail["datetime"] <= end)].copy()
    curt = curt[(curt["datetime"] >= start) & (curt["datetime"] <= end)].copy()

    # -----------------------------
    # 5) merge (handle region collision)
    # -----------------------------
    df = curt.merge(
        avail[["datetime", "node_id", "region", "node_rated_mw", "available_mwh", "x"]],
        on=["datetime", "node_id"],
        how="inner",
        suffixes=("_curt", "_gen"),
    )
    if df.empty:
        raise ValueError("Merge produced 0 rows. Likely datetime alignment or node_id mismatch.")

    # Use generator-derived region
    if "region_gen" in df.columns:
        df["region"] = df["region_gen"]
    elif "region_y" in df.columns:
        df["region"] = df["region_y"]
    df["region"] = df["region"].astype(str).str.strip().str.upper()

    # Response variables
    df["y_avail"] = (df["curtailment_mwh"] / (df["available_mwh"] + EPS_AVAIL)).clip(0, 1)

    # bins
    edges, labels = build_bins(BIN_WIDTH_PCT, INCLUDE_OVER_100)
    df["bin"] = pd.cut(df["x"], bins=edges, labels=labels, include_lowest=True, right=True)

    # local flag
    df["is_local"] = df["region"].isin(LOCAL_REGIONS)

    # =========================
    # CHECKS (simple + useful)
    # =========================
    print("\n=== SANITY CHECKS ===")
    print("[check] df rows:", len(df), "nodes:", df["node_id"].nunique())
    print("[check] datetime:", df["datetime"].min(), "->", df["datetime"].max())
    print("[check] local rows:", int(df["is_local"].sum()), "local nodes:", df.loc[df["is_local"], "node_id"].nunique())
    print("[check] x max:", float(df["x"].max()), "share x>1:", float((df["x"] > 1).mean()))
    print("[check] share x>=0.999:", float((df["x"] >= 0.999).mean()), "share x==1.0:", float((df["x"] == 1.0).mean()))
    print("[check] y_avail min/max:", float(df["y_avail"].min()), float(df["y_avail"].max()),
          "share y_avail>1:", float((df["y_avail"] > 1).mean()))
    # spot check one random row
    r = df.sample(1, random_state=0).iloc[0]
    print("[check] sample row:",
          "node", r["node_id"], "dt", r["datetime"],
          "rated", round(float(r["node_rated_mw"]), 3),
          "avail", round(float(r["available_mwh"]), 3),
          "x", round(float(r["x"]), 4),
          "curt", round(float(r["curtailment_mwh"]), 3),
          "y_avail", round(float(r["y_avail"]), 4))

    # -----------------------------
    # 6) build curves
    # -----------------------------
    nat_curve = agg_curve(df, bin_col="bin", weight_col="node_rated_mw")
    loc_curve = agg_curve(df[df["is_local"]], bin_col="bin", weight_col="node_rated_mw")

    out = nat_curve.merge(loc_curve, on="bin", how="left", suffixes=("_nat", "_loc"))

    # robust shrinkage: fallback to national if local missing in any bin
    loc_ros = out["ros_y_avail_loc"].fillna(out["ros_y_avail_nat"])
    loc_n = out["n_rows_loc"].fillna(0)

    out["y_avail_final"], out["w_shrink"] = shrink(
        local=loc_ros,
        nat=out["ros_y_avail_nat"],
        n_local=loc_n,
        k=float(SHRINK_K),
    )

    apply_table = out[[
        "bin",
        "y_avail_final",
        "w_shrink",
        "ros_y_avail_nat",
        "ros_y_avail_loc",
        "n_rows_nat",
        "n_rows_loc",
        "n_nodes_nat",
        "n_nodes_loc",
        "p10_y_avail_nat",
        "p50_y_avail_nat",
        "p90_y_avail_nat",
        "p10_y_avail_loc",
        "p50_y_avail_loc",
        "p90_y_avail_loc",
    ]].copy()

    apply_table["y_avail_final_pct"] = 100 * apply_table["y_avail_final"]
    apply_table["bin_sort"] = apply_table["bin"].astype(str)
    apply_table = apply_table.sort_values("bin_sort").drop(columns=["bin_sort"]).reset_index(drop=True)

    # -----------------------------
    # 7) export
    # -----------------------------
    xlsx_path = OUT_DIR / "curtailment_propensity_curve_bins_p3.xlsx"
    csv_path = OUT_DIR / "curtailment_propensity_curve_bins_p3.csv"
    pq_path = OUT_DIR / "curtailment_propensity_curve_bins_p3.parquet"

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        apply_table.to_excel(writer, sheet_name="curve_bins", index=False)
        out.to_excel(writer, sheet_name="debug_full", index=False)

    apply_table.to_csv(csv_path, index=False)
    apply_table.to_parquet(pq_path, index=False)

    print("\n[ok] Wrote:", xlsx_path)
    print("[ok] Wrote:", csv_path)
    print("[ok] Wrote:", pq_path)
    print("[info] Apply to new farm: curtailment_hat_mwh = available_mwh_new * y_avail_final (bin-mapped)")


if __name__ == "__main__":
    main()
