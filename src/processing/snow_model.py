import numpy as np
from numba import njit
from typing import Tuple
import xarray as xr


@njit(parallel=False)
def snow_model(
    precip: np.ndarray, t_mean: np.ndarray, days: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the snow based on a simple temperature based model as described
    in Massmann (2019) "Modelling snow in ungauged catchments" with fixed values
    for the snow melt threshold and the degree day factor
    """
    # Degree-day factor [mm / (°C * day)] controlling melt rate per degree above melt temperature
    degree_day = 4.0
    # temperature threshold where precipitation is considered to fall as snow [°C]
    t_thresh = 1.7
    # temperature threshold where snow starts to melt [°C]
    t_melt = 0.0

    # Get the dimensions of the input arrays
    n = precip.shape[0]
    ny, nx = precip.shape[1], precip.shape[2]

    # Initialize output arrays for snow storage, snow melt, and rain
    snow_storage = np.zeros((n, ny, nx), dtype=np.float32)
    snow_melt = np.zeros((n, ny, nx), dtype=np.float32)
    rain = np.zeros((n, ny, nx), dtype=np.float32)

    for month in range(n):
        # rain is everything that is not snow
        mask_snow = t_mean[month] <= t_thresh
        rain[month] = np.where(mask_snow, 0.0, precip[month])
        # no melt in first month, only snow accumulation
        if month == 0:
            snow_storage[month] = np.where(mask_snow, precip[month], 0.0)
            snow_melt[month] = 0.0
        else:
            # Calculate snow melt based on degree-day method
            mask_melt = t_mean[month] >= t_melt
            melt = np.where(
                mask_melt, days[month] * degree_day * (t_mean[month] - t_melt), 0.0
            )
            # melt cannot exceed the available snow storage from the previous month
            melt = np.minimum(melt, snow_storage[month - 1])
            snow_melt[month] = melt
            # new snow for the current month
            new_snow = np.where(mask_snow, precip[month], 0.0)
            # Update snow storage
            snow_storage[month] = snow_storage[month - 1] + new_snow - melt
            snow_storage[month] = np.where(
                snow_storage[month] < 0.0, 0.0, snow_storage[month]
            )

    return snow_storage, snow_melt, rain


def add_snow_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds snow storage, snowmelt and rain variables to the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset must contain: precip_mm_month, t_mean_celsius, days_in_month

    Returns
    -------
    xr.Dataset
        Dataset with added variables: snow_storage, snow_melt
    """
    # Extract necessary variables from the dataset
    precip = ds["precip_mm_month"].values
    t_mean = ds["t_mean_celsius"].values
    days = ds["days_in_month"].values

    # Calculate snow storage, snow melt, and rain using the snow model
    snow_storage, snow_melt, rain = snow_model(precip, t_mean, days)

    # Add the calculated variables back to the dataset with appropriate attributes
    ds["snow_storage"] = (("time", "lat", "lon"), snow_storage)
    ds["snow_storage"].attrs = {
        "units": "mm",
        "description": "Snow water equivalent snow storage",
        "long_name": "Snow storage",
    }

    ds["snow_melt"] = (("time", "lat", "lon"), snow_melt)
    ds["snow_melt"].attrs = {
        "units": "mm month-1",
        "description": "Snow Water equivalent snowmelt",
        "long_name": "Snowmelt",
    }

    ds["rain"] = (("time", "lat", "lon"), rain)
    ds["rain"].attrs = {
        "units": "mm month-1",
        "description": "Monthly rain",
        "long_name": "Rain",
    }

    return ds
