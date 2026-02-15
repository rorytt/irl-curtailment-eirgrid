# src/curtailment/wind_counterfactual.py
from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd


def month_start(dt: pd.Series) -> pd.Series:
    return dt.dt.to_period("M").dt.to_timestamp()


def build_counterfactual_wind(
    df_avail_regions: pd.DataFrame,   # weather_year, datetime, region, technology, available_mwh
    df_avail_nodes: pd.DataFrame,     # weather_year, datetime, node_id, technology, available_mwh
    df_dd_nat: pd.DataFrame,          # weather_year, datetime, dd_mwh
    df_avail_rec_nat: pd.DataFrame,   # weather_year, datetime, avail_den_mwh
    df_dd_pct: pd.DataFrame,          # weather_year, month, region, dd_pct
    alpha_region: float = 2.0,
    beta_node: float = 1.0,
    gate_quantile: float | None = 0.05,
    use_raking: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # --- Filter wind ---
    A_r = df_avail_regions[df_avail_regions["technology"].str.lower().eq("wind")].copy()
    A_n = df_avail_nodes[df_avail_nodes["technology"].str.lower().eq("wind")].copy()

    # Ensure datetime types
    for df in (A_r, A_n, df_dd_nat, df_avail_rec_nat):
        df["datetime"] = pd.to_datetime(df["datetime"])
    df_dd_pct["month"] = pd.to_datetime(df_dd_pct["month"])

    # Rename availability column to standard
    A_r = A_r.rename(columns={"available_mwh": "A_rh"})
    A_n = A_n.rename(columns={"available_mwh": "A_nh"})

    # --- National model availability (sum regions) ---
    A_nat_h = (
        A_r.groupby(["weather_year", "datetime"], as_index=False)["A_rh"]
        .sum()
        .rename(columns={"A_rh": "A_nath"})
    )

    # --- Stage 0: monthly intensity from recorded series ---
    dd_nat = df_dd_nat.copy()
    avail_rec = df_avail_rec_nat.copy()

    dd_nat["month"] = month_start(dd_nat["datetime"])
    avail_rec["month"] = month_start(avail_rec["datetime"])

    DD_rec_m = dd_nat.groupby(["weather_year", "month"], as_index=False)["dd_mwh"].sum().rename(columns={"dd_mwh": "DD_rec_m"})
    A_rec_m = avail_rec.groupby(["weather_year", "month"], as_index=False)["avail_den_mwh"].sum().rename(columns={"avail_den_mwh": "A_rec_m"})

    lam = DD_rec_m.merge(A_rec_m, on=["weather_year", "month"], how="inner")
    lam["lambda_my"] = np.where(lam["A_rec_m"] > 0, lam["DD_rec_m"] / lam["A_rec_m"], 0.0)

    # Model national monthly availability
    A_nat_m = (
        A_nat_h.assign(month=month_start(A_nat_h["datetime"]))
        .groupby(["weather_year", "month"], as_index=False)["A_nath"]
        .sum()
        .rename(columns={"A_nath": "A_nat_m"})
    )

    cf_m = lam.merge(A_nat_m, on=["weather_year", "month"], how="inner")
    cf_m["DD_cf_m"] = cf_m["lambda_my"] * cf_m["A_nat_m"]

    # --- Stage 1: within-month timing shape from recorded national DD ---
    dd_nat = dd_nat.merge(DD_rec_m, on=["weather_year", "month"], how="left")
    dd_nat["shape_h"] = np.where(dd_nat["DD_rec_m"] > 0, dd_nat["dd_mwh"] / dd_nat["DD_rec_m"], 0.0)

    dd_nat = dd_nat.merge(cf_m[["weather_year", "month", "DD_cf_m"]], on=["weather_year", "month"], how="left")
    dd_nat["DD_cf_h"] = dd_nat["DD_cf_m"] * dd_nat["shape_h"]

    # Feasibility cap at national availability
    dd_nat = dd_nat.merge(A_nat_h, on=["weather_year", "datetime"], how="left")
    dd_nat["DD_cf_h"] = np.minimum(dd_nat["DD_cf_h"], dd_nat["A_nath"].fillna(0.0))
    dd_h = dd_nat[["weather_year", "datetime", "DD_cf_h"]].copy()

    # --- Stage 2: hourly split to regions using dd_pct * (A_rh)^alpha ---
    A_r["month"] = month_start(A_r["datetime"])
    w = A_r.merge(df_dd_pct, on=["weather_year", "month", "region"], how="left")
    w["dd_pct"] = w["dd_pct"].fillna(0.0)

    # gating
    if gate_quantile is not None:
        thresh = (
            w.groupby(["weather_year", "month", "region"])["A_rh"]
            .quantile(gate_quantile)
            .reset_index()
            .rename(columns={"A_rh": "gate"})
        )
        w = w.merge(thresh, on=["weather_year", "month", "region"], how="left")
        w["A_eff"] = np.where(w["A_rh"] >= w["gate"], w["A_rh"], 0.0)
    else:
        w["A_eff"] = w["A_rh"]

    w["u_rh"] = w["dd_pct"] * np.power(w["A_eff"], alpha_region)
    w = w.merge(dd_h, on=["weather_year", "datetime"], how="left")

    denom = w.groupby(["weather_year", "datetime"])["u_rh"].transform("sum")
    w["c_rh0"] = np.where(denom > 0, w["DD_cf_h"] * (w["u_rh"] / denom), 0.0)

    # regional cap
    w["c_rh0"] = np.minimum(w["c_rh0"], w["A_rh"])

    df_region_hour = w[["weather_year", "datetime", "region", "month", "c_rh0", "A_rh"]].rename(columns={"c_rh0": "c_mwh"})

    # Optional raking to match scaled regional monthly targets exactly
    if use_raking:
        from .raking import rake_region_hour

        A_rm = A_r.groupby(["weather_year", "month", "region"], as_index=False)["A_rh"].sum().rename(columns={"A_rh": "A_rm"})
        C_raw = df_dd_pct.merge(A_rm, on=["weather_year", "month", "region"], how="left")
        C_raw["C_raw"] = C_raw["dd_pct"].fillna(0.0) * C_raw["A_rm"].fillna(0.0)
        C_raw = C_raw.merge(cf_m[["weather_year", "month", "DD_cf_m"]], on=["weather_year", "month"], how="left")

        sum_raw = C_raw.groupby(["weather_year", "month"])["C_raw"].transform("sum")
        C_raw["C_rm"] = np.where(sum_raw > 0, C_raw["C_raw"] * (C_raw["DD_cf_m"] / sum_raw), 0.0)
        targets = C_raw[["weather_year", "month", "region", "C_rm"]]

        df_region_hour = rake_region_hour(
            df_region_hour=df_region_hour,
            dd_nat_hour=dd_h,
            targets_rm=targets,
            cap_col="A_rh",
        )
    # diagnostics
    tmp = df_region_hour.copy()
    tmp_sum_h = (
        tmp.groupby(["weather_year", "datetime"])["c_mwh"]
        .sum()
        .reset_index(name="sum_reg")
    )

    tmp_check = (
        dd_h.merge(tmp_sum_h, on=["weather_year", "datetime"], how="left")
        .fillna(0.0)
    )

    bad = (tmp_check["DD_cf_h"] > 1e-6) & (tmp_check["sum_reg"] < 1e-6)

    print(f"[diag] hours with DD_cf_h>0 but allocated sum==0: {bad.sum()}")

    if bad.any():
        bad_hours_df = tmp_check.loc[bad, ["weather_year", "datetime", "DD_cf_h", "sum_reg"]].copy()
        bad_hours_df = bad_hours_df.sort_values(["weather_year", "datetime"])

        # print first N (or all)
        print("\n[diag] Bad hours (DD_cf_h>0 but sum allocated ==0):")
        print(bad_hours_df.head(50).to_string(index=False))

        # save for inspection
        out_path = Path("data/processed/curtailment_counterfactual_wind/diag_bad_hours_2025.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bad_hours_df.to_csv(out_path, index=False)
        print(f"[diag] Saved bad hours to: {out_path}")



    # --- Stage 3: region-hour -> node-hour ---
    # Node availability already includes region via nodes.csv merge in IO
    reg = df_region_hour[["weather_year", "datetime", "region", "c_mwh"]].copy()
    nh = A_n.merge(reg, on=["weather_year", "datetime", "region"], how="left")

    nh["v_nh"] = np.power(nh["A_nh"], beta_node)
    denom_n = nh.groupby(["weather_year", "datetime", "region"])["v_nh"].transform("sum")
    nh["c_nh"] = np.where(denom_n > 0, nh["c_mwh"] * (nh["v_nh"] / denom_n), 0.0)

    nh["c_nh"] = np.minimum(nh["c_nh"], nh["A_nh"])

    df_node_hour = nh[["weather_year", "datetime", "region", "node_id", "c_nh", "A_nh"]].rename(
        columns={"c_nh": "c_mwh"}
    )


    df_meta_month = cf_m.copy()

    return (
        df_region_hour[["weather_year", "datetime", "region", "c_mwh"]],
        df_node_hour,
        df_meta_month,
    )
