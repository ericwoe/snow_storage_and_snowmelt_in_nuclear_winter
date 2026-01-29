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
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate snow storage and melt using a degree-day approach.

    Physical model:
    - Snow accumulates when temperature <= threshold
    - Snow melts when temperature >= melt temperature
    - Melt rate is proportional to positive degree-days

    Parameters
    ----------
    precip : np.ndarray, shape (time, lat, lon)
        Precipitation in mm/month
    t_mean : np.ndarray, shape (time, lat, lon)
        Mean monthly temperature in °C
    days : np.ndarray, shape (time,)
        Days per month

    Returns
    -------
    snow_storage : np.ndarray, shape (time, lat, lon)
        Snow water equivalent storage in mm
    snow_melt : np.ndarray, shape (time, lat, lon)
        Monthly snow melt in mm
    """
    degree_day: float = 4.0
    t_threshold: float = 1.7
    t_melt: float = 0.0

    n_time = precip.shape[0]
    n_lat, n_lon = precip.shape[1], precip.shape[2]

    # Initialize output arrays
    snow_storage = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)
    snow_melt = np.zeros((n_time, n_lat, n_lon), dtype=np.float32)

    # Loop over time steps
    for month in range(n_time):
        if month == 0:
            # First month: only accumulation, no melt
            mask_snow = t_mean[month] <= t_threshold
            snow_storage[month] = np.where(mask_snow, precip[month], 0.0)
            snow_melt[month] = 0.0

        else:
            # Calculate potential melt
            mask_melt = t_mean[month] >= t_melt
            potential_melt = np.where(
                mask_melt, days[month] * degree_day * (t_mean[month] - t_melt), 0.0
            )

            # Actual melt cannot exceed available storage
            actual_melt = np.minimum(potential_melt, snow_storage[month - 1])
            snow_melt[month] = actual_melt

            # Calculate new accumulation
            mask_snow = t_mean[month] <= t_threshold
            new_snow = np.where(mask_snow, precip[month], 0.0)

            # Update storage: previous + new snow - melt
            snow_storage[month] = snow_storage[month - 1] + new_snow - actual_melt

            # Prevent negative storage (shouldn't happen, but safety check)
            snow_storage[month] = np.where(
                snow_storage[month] < 0.0, 0.0, snow_storage[month]
            )

    # Round results to 1 decimal place
    snow_storage = np.round(snow_storage, 1)
    snow_melt = np.round(snow_melt, 1)

    return snow_storage, snow_melt


if __name__ == "__main__":
    dataset = xr.open_dataset(
        "../data/interim/preprocessed/climate_data_processed.nc", engine="netcdf4"
    )
    snow_storage, snow_melt = calculate_snow_dynamics(
        precip=dataset["precip_mm_month"].values,
        t_mean=dataset["t_mean_celsius"].values,
        days=dataset["days_in_month"].values,
    )
    dataset["snow_storage_mm"] = (("time", "lat", "lon"), snow_storage)
    dataset["snow_melt_mm"] = (("time", "lat", "lon"), snow_melt)
    dataset.to_netcdf("../results/snow_model_output.nc", engine="netcdf4")
