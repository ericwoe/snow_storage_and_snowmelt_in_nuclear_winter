"""
Snow accumulation and melt model using degree-day approach
"""

import numpy as np
from numba import njit
from typing import Tuple
import xarray as xr


@njit(parallel=False)
def calculate_snow_dynamics(
    precip: np.ndarray, t_mean: np.ndarray, days: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    n = precip.shape[0]
    ny, nx = precip.shape[1], precip.shape[2]

    snow_storage = np.zeros((n, ny, nx), dtype=np.float32)
    snow_melt = np.zeros((n, ny, nx), dtype=np.float32)
    rainfall = np.zeros((n, ny, nx), dtype=np.float32)  # NEU

    degree_day = 4.0
    t_tresh = 1.7
    t_melt = 0.0

    for month in range(n):
        # Regen ist alles was nicht als Schnee fällt
        mask_snow = t_mean[month] <= t_tresh
        rainfall[month] = np.where(mask_snow, 0.0, precip[month])  # NEU

        if month == 0:
            snow_storage[month] = np.where(mask_snow, precip[month], 0.0)
            snow_melt[month] = 0.0
        else:
            mask_melt = t_mean[month] >= t_melt
            melt = np.where(
                mask_melt, days[month] * degree_day * (t_mean[month] - t_melt), 0.0
            )
            melt = np.minimum(melt, snow_storage[month - 1])
            snow_melt[month] = melt

            new_snow = np.where(mask_snow, precip[month], 0.0)
            snow_storage[month] = snow_storage[month - 1] + new_snow - melt
            snow_storage[month] = np.where(
                snow_storage[month] < 0.0, 0.0, snow_storage[month]
            )

    return snow_storage, snow_melt, rainfall


def add_snow_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Add snow storage and melt variables to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset must contain: precip_mm_month, t_mean_celsius, days_in_month

    Returns
    -------
    xr.Dataset
        Dataset with added variables: snow_storage, snow_melt
    """
    # Extrahiere NumPy-Arrays
    precip = ds["precip_mm_month"].values
    t_mean = ds["t_mean_celsius"].values
    days = ds["days_in_month"].values

    # Führe Berechnung aus
    snow_storage, snow_melt, rainfall = calculate_snow_dynamics(precip, t_mean, days)

    # Füge als xarray-Variablen hinzu (mit Koordinaten!)
    ds["snow_storage"] = (("time", "lat", "lon"), snow_storage)
    ds["snow_storage"].attrs = {
        "units": "mm",
        "description": "Snow water equivalent storage",
        "long_name": "Snow storage",
    }

    ds["snow_melt"] = (("time", "lat", "lon"), snow_melt)
    ds["snow_melt"].attrs = {
        "units": "mm month-1",
        "description": "Monthly snowmelt",
        "long_name": "Snow melt",
    }

    ds["rainfall"] = (("time", "lat", "lon"), rainfall)
    ds["rainfall"].attrs = {
        "units": "mm month-1",
        "description": "Monthly rainfall",
        "long_name": "Rainfall",
    }

    return ds
