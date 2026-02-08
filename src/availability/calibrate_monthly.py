# src/availability/calibrate_monthly.py
from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


# ---------------------------
# IO + normalisation helpers
# ---------------------------

def _read_recorded_monthly(csv_path: Path) -> pd.DataFrame:
    """
    Expected minimum columns:
      year (int), month (int 1-12), technology (wind/solar), recorded_gwh (float)

    Allows common alternative names and normalises.
    """
    rec = pd.read_csv(csv_path)
    rec.columns = [c.strip().lower() for c in rec.columns]

    # Column aliases (lightweight)
    col_map = {}
    if "recorded_gwh" not in rec.columns:
        for alt in ["recorded", "gwh", "recorded_total_gwh", "monthly_gwh", "availability_gwh"]:
            if alt in rec.columns:
                col_map[alt] = "recorded_gwh"
                break
    if "technology" not in rec.columns:
        for alt in ["tech", "fuel", "type"]:
            if alt in rec.columns:
                col_map[alt] = "technology"
                break
    if col_map:
        rec = rec.rename(columns=col_map)

    required = {"year", "month", "technology", "recorded_gwh"}
    missing = required - set(rec.columns)
    if missing:
        raise ValueError(f"Recorded monthly CSV missing columns: {missing}. Found: {list(rec.columns)}")

    rec["technology"] = rec["technology"].astype(str).str.strip().str.lower()
    rec = rec[rec["technology"].isin(["wind", "solar"])].copy()

    rec["year"] = rec["year"].astype(int)
    rec["month"] = rec["month"].astype(int)
    if (~rec["month"].between(1, 12)).any():
        bad = sorted(rec.loc[~rec["month"].between(1, 12), "month"].unique().tolist())
        raise ValueError(f"Invalid month values in recorded CSV: {bad}")

    rec["recorded_gwh"] = rec["recorded_gwh"].astype(float)

    # Aggregate in case file has duplicates
    rec = (
        rec.groupby(["year", "month", "technology"], as_index=False)
        .agg(recorded_gwh=("recorded_gwh", "sum"))
        .sort_values(["technology", "year", "month"])
        .reset_index(drop=True)
    )
    return rec


def _model_monthly_from_region(av_reg: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly model totals (GWh) by technology from region-hourly availability.
    availability_mwh is hourly energy; monthly sum /1000 = GWh.
    """
    df = av_reg.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()

    # system total per hour (sum regions)
    sys_hour = (
        df.groupby(["datetime", "technology"], as_index=False)
        .agg(available_mwh=("available_mwh", "sum"))
    )
    sys_hour["year"] = sys_hour["datetime"].dt.year.astype(int)
    sys_hour["month"] = sys_hour["datetime"].dt.month.astype(int)

    monthly = (
        sys_hour.groupby(["year", "month", "technology"], as_index=False)
        .agg(model_gwh=("available_mwh", lambda s: float(s.sum() / 1e3)))
        .sort_values(["technology", "year", "month"])
        .reset_index(drop=True)
    )
    return monthly


# ---------------------------
# Factor building + smoothing
# ---------------------------

def _smooth_factors_12mo(factors: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Smooth factors across months within each technology (circular rolling mean).
    Expects columns: technology, month, k
    window must be odd. window=1 => no smoothing.
    """
    if window <= 1:
        factors["k_smooth"] = factors["k"]
        return factors

    if window % 2 == 0:
        raise ValueError("Smoothing window must be odd (e.g., 3 or 5).")

    out = []
    for tech, g in factors.groupby("technology", sort=False):
        g = g.sort_values("month").copy()
        if g["month"].nunique() != 12:
            # fallback non-circular
            g["k_smooth"] = g["k"].rolling(window=window, center=True, min_periods=1).mean()
            out.append(g)
            continue

        k = g["k"].to_numpy(dtype=float)
        pad = window // 2
        k_pad = np.concatenate([k[-pad:], k, k[:pad]])
        k_smooth = pd.Series(k_pad).rolling(window=window, center=True).mean().to_numpy()
        g["k_smooth"] = k_smooth[pad : pad + 12]
        out.append(g)

    return pd.concat(out, ignore_index=True)


def _build_multi_year_factors(
    years: list[int],
    recorded: pd.DataFrame,
    model_monthly_by_year: dict[int, pd.DataFrame],
    combine: str = "weighted",
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    smooth_window: int = 3,
    use_smooth: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - factors_month: columns [month, technology, k_raw, k_smooth] (one set to apply to any year)
      - diagnostics: per-year per-month table with recorded/model/k_raw for transparency
    combine:
      - "weighted": weighted average of per-year ratios using recorded_gwh as weights
      - "mean": simple average of per-year ratios
    """
    diagnostics = []

    # Build per-year ratios
    for y in years:
        rec_y = recorded[recorded["year"] == y].copy()
        if rec_y.empty:
            raise ValueError(f"Recorded CSV has no rows for year {y}")

        mod_y = model_monthly_by_year.get(y)
        if mod_y is None or mod_y.empty:
            raise ValueError(f"Model monthly table missing for year {y}")

        comp = rec_y.merge(mod_y, on=["year", "month", "technology"], how="left")
        if comp["model_gwh"].isna().any():
            miss = comp[comp["model_gwh"].isna()][["month", "technology"]]
            raise ValueError(f"Model missing some month/tech for year {y}:\n{miss.to_string(index=False)}")

        comp["k_raw_year"] = comp["recorded_gwh"] / comp["model_gwh"]
        diagnostics.append(comp[["year", "month", "technology", "recorded_gwh", "model_gwh", "k_raw_year"]])

    diag = pd.concat(diagnostics, ignore_index=True)

    # Combine across years into a single 12-month factor per tech
    if combine not in {"weighted", "mean"}:
        raise ValueError("combine must be one of: weighted, mean")

    if combine == "weighted":
        # weighted by recorded_gwh (stable and intuitive)
        def _wavg(g: pd.DataFrame) -> float:
            w = g["recorded_gwh"].to_numpy(dtype=float)
            x = g["k_raw_year"].to_numpy(dtype=float)
            denom = w.sum()
            return float((w * x).sum() / denom) if denom > 0 else float(np.nan)

        k_month = (
            diag.groupby(["month", "technology"], as_index=False)
            .apply(lambda g: pd.Series({"k": _wavg(g)}))
            .reset_index(drop=True)
        )
    else:
        k_month = (
            diag.groupby(["month", "technology"], as_index=False)
            .agg(k=("k_raw_year", "mean"))
        )

    # Optional clamp to stop any month being extreme
    if clamp_min is not None or clamp_max is not None:
        lo = -np.inf if clamp_min is None else float(clamp_min)
        hi = np.inf if clamp_max is None else float(clamp_max)
        k_month["k"] = k_month["k"].clip(lower=lo, upper=hi)

    # Smooth across months (per tech)
    k_month = _smooth_factors_12mo(k_month, window=smooth_window)
    if not use_smooth:
        k_month["k_smooth"] = k_month["k"]

    # Pretty print order
    k_month = k_month.sort_values(["technology", "month"]).reset_index(drop=True)
    return k_month[["month", "technology", "k", "k_smooth"]], diag.sort_values(["technology", "year", "month"])


# ---------------------------
# Apply factors
# ---------------------------

def _apply_monthly_factors_any_year(av: pd.DataFrame, year: int, factors_month: pd.DataFrame, use_smooth: bool = True) -> pd.DataFrame:
    """
    Apply factors defined by (month, technology) to an availability dataframe for a specific year.
    Expects columns: datetime, technology, available_mwh plus node_id/region.
    """
    df = av.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df["technology"] = df["technology"].astype(str).str.strip().str.lower()
    df = df[df["technology"].isin(["wind", "solar"])].copy()

    df["year"] = df["datetime"].dt.year.astype(int)
    df["month"] = df["datetime"].dt.month.astype(int)

    # ensure we're applying to the requested year only
    df = df[df["year"] == year].copy()

    key = "k_smooth" if use_smooth else "k"
    f = factors_month[["month", "technology", key]].rename(columns={key: "k"}).copy()

    df = df.merge(f, on=["month", "technology"], how="left")
    if df["k"].isna().any():
        miss = df[df["k"].isna()][["month", "technology"]].drop_duplicates()
        raise ValueError(f"Missing factors for some month/tech:\n{miss.to_string(index=False)}")

    df["available_mwh"] = df["available_mwh"] * df["k"]
    df = df.drop(columns=["year", "month", "k"])
    return df


def _compare_monthly_original_vs_scaled(av_reg_original: pd.DataFrame, av_reg_scaled: pd.DataFrame, year: int) -> pd.DataFrame:
    orig = _model_monthly_from_region(av_reg_original).rename(columns={"model_gwh": "original_gwh"})
    scal = _model_monthly_from_region(av_reg_scaled).rename(columns={"model_gwh": "scaled_gwh"})

    orig = orig[orig["year"] == year].copy()
    scal = scal[scal["year"] == year].copy()

    comp = orig.merge(scal, on=["year", "month", "technology"], how="inner")
    comp["delta_gwh"] = comp["scaled_gwh"] - comp["original_gwh"]
    comp["delta_pct"] = np.where(comp["original_gwh"] != 0, comp["delta_gwh"] / comp["original_gwh"] * 100.0, np.nan)
    return comp.sort_values(["technology", "month"]).reset_index(drop=True)


# ---------------------------
# Main calibrator
# ---------------------------

def calibrate_monthly_multi_base(
    base_years: list[int],
    apply_years: list[int],
    recorded_csv: Path,
    combine: str = "weighted",
    smooth_window: int = 3,
    use_smooth: bool = True,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> None:
    recorded = _read_recorded_monthly(recorded_csv)

    # Load model monthly for each base year from region availability files
    model_monthly_by_year: dict[int, pd.DataFrame] = {}
    for y in base_years:
        reg_path = Path(f"data/processed/availability_region_hourly/availability_regions_{y}.parquet")
        if not reg_path.exists():
            raise FileNotFoundError(f"Missing base-year region availability: {reg_path} (run build_availability)")

        av_reg = pd.read_parquet(reg_path)
        model_monthly = _model_monthly_from_region(av_reg)
        model_monthly_by_year[y] = model_monthly[model_monthly["year"] == y].copy()

    # Build factors from base years
    factors_month, diag = _build_multi_year_factors(
        years=base_years,
        recorded=recorded,
        model_monthly_by_year=model_monthly_by_year,
        combine=combine,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        smooth_window=smooth_window,
        use_smooth=use_smooth,
    )

    # Print per-year diagnostics (recorded vs model ratios)
    print("=== Base-year diagnostics (recorded vs model, per month) ===")
    view = diag.copy()
    view["model_gwh"] = view["model_gwh"].map(lambda x: f"{x:.1f}")
    view["recorded_gwh"] = view["recorded_gwh"].map(lambda x: f"{x:.1f}")
    view["k_raw_year"] = view["k_raw_year"].map(lambda x: f"{x:.3f}")
    print(view[["year", "technology", "month", "model_gwh", "recorded_gwh", "k_raw_year"]].to_string(index=False))

    # Print combined factors
    print("\n=== Combined monthly factors (applied to all years) ===")
    show = factors_month.copy()
    show["k"] = show["k"].map(lambda x: f"{x:.3f}")
    show["k_smooth"] = show["k_smooth"].map(lambda x: f"{x:.3f}")
    print(show.to_string(index=False))

    # Output dirs
    out_node_dir = Path("data/processed/availability_node_hourly_calibrated")
    out_reg_dir = Path("data/processed/availability_region_hourly_calibrated")
    out_node_dir.mkdir(parents=True, exist_ok=True)
    out_reg_dir.mkdir(parents=True, exist_ok=True)

    # Apply to requested years
    for y in apply_years:
        reg_in = Path(f"data/processed/availability_region_hourly/availability_regions_{y}.parquet")
        node_in = Path(f"data/processed/availability_node_hourly/availability_nodes_{y}.parquet")
        if not reg_in.exists() or not node_in.exists():
            raise FileNotFoundError(f"Missing availability for year {y}: {reg_in} or {node_in} (run build_availability)")

        av_reg = pd.read_parquet(reg_in)
        av_node = pd.read_parquet(node_in)

        reg_scaled = _apply_monthly_factors_any_year(av_reg, y, factors_month, use_smooth=use_smooth)
        node_scaled = _apply_monthly_factors_any_year(av_node, y, factors_month, use_smooth=use_smooth)

        out_reg = out_reg_dir / f"availability_regions_{y}_calibrated.parquet"
        out_node = out_node_dir / f"availability_nodes_{y}_calibrated.parquet"
        reg_scaled.to_parquet(out_reg, index=False)
        node_scaled.to_parquet(out_node, index=False)

        print(f"\nWrote {out_reg}")
        print(f"Wrote {out_node}")

        # For non-recorded years, compare original vs scaled totals
        comp = _compare_monthly_original_vs_scaled(av_reg, reg_scaled, year=y)
        print(f"\n=== Year {y} monthly totals: ORIGINAL vs SCALED (system) ===")
        viewc = comp.copy()
        viewc["original_gwh"] = viewc["original_gwh"].map(lambda x: f"{x:.1f}")
        viewc["scaled_gwh"] = viewc["scaled_gwh"].map(lambda x: f"{x:.1f}")
        viewc["delta_gwh"] = viewc["delta_gwh"].map(lambda x: f"{x:+.1f}")
        viewc["delta_pct"] = viewc["delta_pct"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "nan")
        print(viewc[["technology", "month", "original_gwh", "scaled_gwh", "delta_gwh", "delta_pct"]].to_string(index=False))

        tot = comp.groupby("technology")[["original_gwh", "scaled_gwh", "delta_gwh"]].sum().reset_index()
        tot["delta_pct"] = np.where(tot["original_gwh"] != 0, tot["delta_gwh"] / tot["original_gwh"] * 100.0, np.nan)
        tot["original_gwh"] = tot["original_gwh"].map(lambda x: f"{x:.1f}")
        tot["scaled_gwh"] = tot["scaled_gwh"].map(lambda x: f"{x:.1f}")
        tot["delta_gwh"] = tot["delta_gwh"].map(lambda x: f"{x:+.1f}")
        tot["delta_pct"] = tot["delta_pct"].map(lambda x: f"{x:+.2f}%" if pd.notna(x) else "nan")
        print(f"\n=== Year {y} totals by tech: ORIGINAL vs SCALED ===")
        print(tot.to_string(index=False))

        # If year y is one of the base years and recorded exists, validate against recorded
        if y in base_years:
            rec_y = recorded[recorded["year"] == y].copy()
            scaled_monthly = _model_monthly_from_region(reg_scaled).rename(columns={"model_gwh": "scaled_gwh"})
            scaled_monthly = scaled_monthly[scaled_monthly["year"] == y].copy()

            chk = rec_y.merge(scaled_monthly, on=["year", "month", "technology"], how="left")
            if chk["scaled_gwh"].isna().any():
                miss = chk[chk["scaled_gwh"].isna()][["month", "technology"]]
                print(f"\nWARNING: could not validate some month/tech for year {y}:\n{miss.to_string(index=False)}")
            else:
                chk["err_pct"] = (chk["scaled_gwh"] - chk["recorded_gwh"]) / chk["recorded_gwh"] * 100.0
                print(f"\n=== Base-year validation (recorded vs scaled) — {y} ===")
                vv = chk.sort_values(["technology", "month"]).copy()
                vv["recorded_gwh"] = vv["recorded_gwh"].map(lambda x: f"{x:.1f}")
                vv["scaled_gwh"] = vv["scaled_gwh"].map(lambda x: f"{x:.1f}")
                vv["err_pct"] = vv["err_pct"].map(lambda x: f"{x:+.1f}%")
                print(vv[["technology", "month", "recorded_gwh", "scaled_gwh", "err_pct"]].to_string(index=False))

                tot2 = chk.groupby("technology")[["recorded_gwh", "scaled_gwh"]].sum().reset_index()
                tot2["err_pct"] = (tot2["scaled_gwh"] - tot2["recorded_gwh"]) / tot2["recorded_gwh"] * 100.0
                tot2["recorded_gwh"] = tot2["recorded_gwh"].map(lambda x: f"{x:.1f}")
                tot2["scaled_gwh"] = tot2["scaled_gwh"].map(lambda x: f"{x:.1f}")
                tot2["err_pct"] = tot2["err_pct"].map(lambda x: f"{x:+.2f}%")
                print(f"\n=== Base-year totals (recorded vs scaled) — {y} ===")
                print(tot2.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Monthly calibration of availability using multiple base years (e.g., 2024+2025).")
    parser.add_argument("--base-years", type=str, default="2024,2025",
                        help="Comma-separated base years to compute calibration factors (default: 2024,2025)")
    parser.add_argument("--apply-years", type=str, default="2024,2025",
                        help="Comma-separated years to apply factors to (e.g. 2022,2023,2024,2025). Default: 2024,2025")
    parser.add_argument("--recorded-csv", type=str, default="data/raw/recorded_monthly_availability.csv",
                        help="Path to recorded monthly CSV (default: data/raw/recorded_monthly_availability.csv)")
    parser.add_argument("--combine", type=str, default="weighted", choices=["weighted", "mean"],
                        help="How to combine base-year ratios into one factor per month/tech (default: weighted)")
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="Odd window for circular smoothing across months (1=no smoothing). Default: 3")
    parser.add_argument("--no-smooth", action="store_true", help="Use unsmoothed monthly factors")
    parser.add_argument("--clamp-min", type=float, default=None, help="Optional minimum factor clamp (e.g. 0.7)")
    parser.add_argument("--clamp-max", type=float, default=None, help="Optional maximum factor clamp (e.g. 1.5)")
    args = parser.parse_args()

    base_years = [int(x.strip()) for x in args.base_years.split(",") if x.strip()]
    apply_years = [int(x.strip()) for x in args.apply_years.split(",") if x.strip()]

    calibrate_monthly_multi_base(
        base_years=base_years,
        apply_years=apply_years,
        recorded_csv=Path(args.recorded_csv),
        combine=args.combine,
        smooth_window=args.smooth_window,
        use_smooth=(not args.no_smooth),
        clamp_min=args.clamp_min,
        clamp_max=args.clamp_max,
    )


if __name__ == "__main__":
    main()
