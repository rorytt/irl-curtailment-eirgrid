# src/cf/solar.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


def solar_cf_pvwatts(
    df: pd.DataFrame,
    lat: float,
    lon: float,
    ssrd_col: str = "ssrd_jm2",
    t2m_col: str = "t2m_k",
    u_col: str | None = "u100_ms",
    v_col: str | None = "v100_ms",
    tilt: float | None = None,
    azimuth: float = 180.0,
    gamma_pdc: float = -0.004,   # PVWatts default temp coeff (1/°C)
) -> pd.Series:
    """
    Improved node-level PV CF from ERA5:
      - SSRD (J/m² per hour) -> GHI (W/m²)
      - Solar position -> zenith
      - Erbs decomposition: GHI -> DNI/DHI
      - POA (plane-of-array) via isotropic sky model
      - Cell temperature via SAPM cell model (needs wind speed; uses 100m wind scaled to 10m)
      - PVWatts DC power -> CF

    Returns CF in [0, 1].
    """
  # time (force a DatetimeIndex)
    times = pd.DatetimeIndex(pd.to_datetime(df["datetime"], utc=True))

    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)
    zenith = solpos["zenith"].to_numpy()

    # GHI in W/m² from SSRD J/m²/hour
    ghi = df[ssrd_col].to_numpy(dtype=float) / 3600.0
    ghi = np.clip(ghi, 0.0, None)

    # temperature
    temp_air = (df[t2m_col].to_numpy(dtype=float) - 273.15)

    # wind speed for temperature model (optional)
    if u_col is not None and v_col is not None and (u_col in df.columns) and (v_col in df.columns):
        v100 = np.sqrt(df[u_col].to_numpy() ** 2 + df[v_col].to_numpy() ** 2)
        # scale 100m -> 10m (power law), alpha ~ 0.14 (neutral-ish)
        wind10 = v100 * (10.0 / 100.0) ** 0.14
        wind10 = np.clip(wind10, 0.0, None)
    else:
        # fallback constant wind if not available
        wind10 = np.full_like(ghi, 1.0, dtype=float)

    # night handling
    night = (zenith >= 90) | (ghi <= 0)
    ghi_n = ghi.copy()
    ghi_n[night] = 0.0

    # Decompose GHI -> DNI/DHI (Erbs)
    # PASS day-of-year explicitly to avoid pandas DatetimeArray arithmetic issues
    doy = times.dayofyear.to_numpy()

    dni_dhi = pvlib.irradiance.erbs(
        ghi=ghi_n,
        zenith=zenith,
        datetime_or_doy=doy,
    )

    # pvlib version-safe extraction
    dni = np.asarray(dni_dhi["dni"])
    dhi = np.asarray(dni_dhi["dhi"])



    # ensure physical non-negatives + enforce night zeros
    dni = np.clip(dni, 0.0, None)
    dhi = np.clip(dhi, 0.0, None)
    dni[night] = 0.0
    dhi[night] = 0.0

    # tilt default = abs(lat)
    if tilt is None:
        tilt = abs(lat)

    # POA irradiance
    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=dni,
        ghi=ghi_n,
        dhi=dhi,
        model="isotropic",
    )["poa_global"].to_numpy()

    poa = np.clip(poa, 0.0, None)

    # Cell temperature (SAPM) requires model parameters in your pvlib version
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"]["open_rack_glass_glass"]
    temp_cell = pvlib.temperature.sapm_cell(
        poa_global=poa,
        temp_air=temp_air,
        wind_speed=wind10,
        a=params["a"],
        b=params["b"],
        deltaT=params["deltaT"],
    )
    temp_cell = np.asarray(temp_cell)


    # PVWatts DC model
    # Use pdc0=1000 W so CF = p_dc / 1000
    p_dc = pvlib.pvsystem.pvwatts_dc(
    effective_irradiance=poa,
    temp_cell=temp_cell,
    pdc0=1000.0,
    gamma_pdc=gamma_pdc,
    )
    cf = (p_dc / 1000.0).astype(float)
    cf[night] = 0.0

    return pd.Series(np.clip(cf, 0.0, 1.0), index=df.index, name="cf")
