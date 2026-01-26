import pandas as pd
import numpy as np
import pvlib

def solar_cf_pvlib(df: pd.DataFrame, lat: float, lon: float) -> pd.Series:
    # datetime must be tz-aware UTC
    times = pd.DatetimeIndex(df["datetime"]).tz_convert("UTC")

    ghi = df["ssrd_jm2"].values / 3600.0  # W/m^2
    temp_c = df["t2m_k"].values - 273.15

    solpos = pvlib.solarposition.get_solarposition(times, lat, lon)

    # Simple fixed tilt
    tilt = abs(lat)
    azimuth = 180.0

    poa = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=None,
        ghi=ghi,
        dhi=None,
        model="isotropic",
    )["poa_global"].clip(lower=0.0)

    # PVWatts-like: power ~ POA/1000 with temp derate
    cf = (poa / 1000.0) * (1.0 + (-0.004) * (temp_c - 25.0))
    cf = np.clip(cf, 0.0, 1.0)

    return pd.Series(cf, index=df.index)
