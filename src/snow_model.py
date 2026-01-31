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
    Calculate snow accumulation and melt using a temperature-index (degree-day) model.

    This function implements a simple snow hydrology model where:
    - Snow accumulates when temperature falls below a threshold
    - Snow melts proportionally to positive degree-days above the melt temperature
    - Melt cannot exceed available snow storage

    The implementation is optimized for Numba JIT compilation using vectorized
    NumPy operations instead of pixel-wise loops.

    Parameters
    ----------
    precip : np.ndarray, shape (time, lat, lon)
        Monthly precipitation in mm/month.
    t_mean : np.ndarray, shape (time, lat, lon)
        Mean monthly temperature in degrees Celsius.
    days : np.ndarray, shape (time,)
        Number of days in each month. Used to convert degree-day factors to
        monthly melt amounts.

    Returns
    -------
    snow_storage : np.ndarray, shape (time, lat, lon)
        Snow water equivalent (SWE) storage at the end of each month in mm.
        Represents the amount of water that would be released if all snow melted.
    snow_melt : np.ndarray, shape (time, lat, lon)
        Monthly snowmelt in mm. This is the amount of water released from the
        snowpack during each month.

    Notes
    -----
    Physical parameters (hardcoded):
    - degree_day = 4.0 mm °C⁻¹ day⁻¹ : Degree-day melt factor
    - t_tresh = 1.7 °C : Temperature threshold for snow accumulation
    - t_melt = 0.0 °C : Base temperature for snowmelt

    Algorithm:
    1. For the first month: precipitation accumulates as snow if T ≤ t_tresh, no melt
    2. For subsequent months:
       a. Calculate potential melt based on positive degree-days
       b. Limit melt to available snow storage from previous month
       c. Add new snow if T ≤ t_tresh
       d. Update storage (previous storage + new snow - melt)
       e. Ensure storage is non-negative

    The function is designed for efficient execution with Numba's @njit decorator
    and uses only NumPy operations compatible with nopython mode.
    """
    n = precip.shape[0]
    ny, nx = precip.shape[1], precip.shape[2]

    # Allocate output arrays
    snow_storage = np.zeros((n, ny, nx), dtype=np.float32)
    snow_melt = np.zeros((n, ny, nx), dtype=np.float32)

    # Physical model parameters
    degree_day = 4.0  # Degree-day melt factor [mm °C⁻¹ day⁻¹]
    t_tresh = 1.7  # Snow accumulation threshold temperature [°C]
    t_melt = 0.0  # Melt base temperature [°C]

    # Loop over time steps
    for month in range(n):
        if month == 0:
            # First month: only accumulation, no melt
            mask_snow = t_mean[month] <= t_tresh
            snow_storage[month] = np.where(mask_snow, precip[month], 0.0)
            snow_melt[month] = 0.0
        else:
            # === SNOWMELT ===
            # Calculate potential melt using degree-day method
            mask_melt = t_mean[month] >= t_melt
            melt = np.where(
                mask_melt, days[month] * degree_day * (t_mean[month] - t_melt), 0.0
            )
            # Constrain melt to available snow storage
            melt = np.minimum(melt, snow_storage[month - 1])
            snow_melt[month] = melt

            # === SNOW ACCUMULATION ===
            mask_snow = t_mean[month] <= t_tresh
            new_snow = np.where(mask_snow, precip[month], 0.0)

            # === UPDATE STORAGE ===
            # Storage = previous storage + new snow - melt
            snow_storage[month] = snow_storage[month - 1] + new_snow - melt

            # Ensure non-negative storage
            snow_storage[month] = np.where(
                snow_storage[month] < 0.0, 0.0, snow_storage[month]
            )

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
