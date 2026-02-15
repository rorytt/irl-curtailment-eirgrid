# src/curtailment/raking.py
from __future__ import annotations
import pandas as pd
import numpy as np

def rake_region_hour(
    df_region_hour: pd.DataFrame,
    dd_nat_hour: pd.DataFrame,      # [weather_year, datetime, DD_cf_h]
    targets_rm: pd.DataFrame,       # [weather_year, month, region, C_rm]
    cap_col: str = "A_rh",
    max_iter: int = 30,
    tol: float = 1e-6,
) -> pd.DataFrame:
    df = df_region_hour.copy()
    df = df.merge(dd_nat_hour, on=["weather_year", "datetime"], how="left")
    df = df.merge(targets_rm, on=["weather_year", "month", "region"], how="left")
    df["C_rm"] = df["C_rm"].fillna(0.0)
    df["DD_cf_h"] = df["DD_cf_h"].fillna(0.0)

    # Ensure numeric arrays
    c = df["c_mwh"].to_numpy(dtype=float)

    for _ in range(max_iter):
        c_prev = c.copy()

        # ---------- (a) month-region scaling ----------
        df["c_mwh"] = c
        sum_rm = df.groupby(["weather_year", "month", "region"])["c_mwh"].transform("sum").to_numpy(dtype=float)
        target_rm = df["C_rm"].to_numpy(dtype=float)

        # Only scale where sum_rm > 0 OR (sum_rm == 0 and target==0) -> keep as 0
        scale_rm = np.ones_like(c)
        mask_rm = sum_rm > 0
        scale_rm[mask_rm] = target_rm[mask_rm] / sum_rm[mask_rm]
        # where sum_rm==0: leave scale=1 (keeps c at 0); targets there should be 0 anyway

        c = c * scale_rm

        # cap
        if cap_col in df.columns:
            cap = df[cap_col].to_numpy(dtype=float)
            c = np.minimum(c, cap)

        # ---------- (b) hourly renormalisation ----------
        df["c_mwh"] = c
        sum_h = df.groupby(["weather_year", "datetime"])["c_mwh"].transform("sum").to_numpy(dtype=float)
        target_h = df["DD_cf_h"].to_numpy(dtype=float)

        scale_h = np.ones_like(c)
        mask_h = sum_h > 0
        scale_h[mask_h] = target_h[mask_h] / sum_h[mask_h]
        # where sum_h==0: keep c at 0; target_h should also be 0 in those hours

        c = c * scale_h

        # cap again
        if cap_col in df.columns:
            cap = df[cap_col].to_numpy(dtype=float)
            c = np.minimum(c, cap)

        # Replace any tiny numerical garbage
        c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)

        if np.max(np.abs(c - c_prev)) < tol:
            break

    df["c_mwh"] = c
    return df.drop(columns=[col for col in ["DD_cf_h", "C_rm"] if col in df.columns])
