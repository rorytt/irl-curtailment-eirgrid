# src/cf/wind.py
from __future__ import annotations
import numpy as np
import pandas as pd


def wind_cf_from_u_v(
    df: pd.DataFrame,
    u_col: str = "u100_ms",
    v_col: str = "v100_ms",
    v_ci: float = 3.0,
    v_r: float = 12.0,
    v_co: float = 25.0,
) -> pd.Series:
    """
    Simple, thesis-friendly wind CF using a generic power curve:
      - 0 below cut-in and above cut-out
      - cubic ramp between cut-in and rated
      - 1 between rated and cut-out
    """
    v = np.sqrt(df[u_col].to_numpy() ** 2 + df[v_col].to_numpy() ** 2)
    cf = np.zeros_like(v, dtype=float)

    ramp = (v >= v_ci) & (v < v_r)
    cf[ramp] = (v[ramp] ** 3 - v_ci**3) / (v_r**3 - v_ci**3)

    rated = (v >= v_r) & (v < v_co)
    cf[rated] = 1.0

    return pd.Series(np.clip(cf, 0.0, 1.0), index=df.index, name="cf")
