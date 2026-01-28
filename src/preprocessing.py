import os
import xarray as xr
from pathlib import Path
import dask
from land_mask import *

def load_climate_files(data_dir, pattern):
    """
    Reads multiple NetCDF files and combines them
    
    Arguments:
        data_dir: Path to directory containing the files
        pattern: Glob pattern for the files (e.g. "*.nc")
    
    Returns:
        xarray.Dataset with combined data
    """
    data_path = Path(data_dir)
    
    # Find and sort files
    files = sorted(data_path.glob(pattern))
    
    # Check if files were found
    if not files:
        raise FileNotFoundError(
            f"No files found in {data_path} with pattern '{pattern}'"
        )
    
    print(f"Found: {len(files)} files")
    print(f"First file: {files[0].name}")
    print(f"Last file: {files[-1].name}")
    
    # Load files
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=False,
        use_cftime=True,
        engine = "netcdf4"
    )
    return ds

def calculate_days_in_month(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculate the number of days in each month from time bounds.
    
    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing time dimension with bounds attribute
    
    Returns
    -------
    xr.DataArray
        DataArray of days per month (integer) with same dimensions as time
    """
    bnds_name = ds.time.attrs["bounds"]
    time_bnds = ds[bnds_name]
    
    # Days in month = end of month - start of month
    days_in_month = time_bnds.isel(nbnd=1) - time_bnds.isel(nbnd=0)
    days_in_month = days_in_month.astype("timedelta64[D]").astype(int)
    
    # Return as DataArray 
    return days_in_month

def calculate_total_precipitation_rate(
    ds: xr.Dataset,
    var_precc: str = "PRECC",
    var_precl: str = "PRECL",
) -> xr.DataArray:
    """
    Calculate total precipitation rate (PRECC + PRECL).

    Returns
    -------
    xr.DataArray
        Total precipitation rate in m s-1
    """
    precip_rate_m_s = ds[var_precc] + ds[var_precl]
    return precip_rate_m_s

def convert_precip_to_mm_month(
    rate: xr.DataArray,
    days_in_month: xr.DataArray,
) -> xr.DataArray:
    """
    Convert precipitation rate from m s-1 to mm month-1.
    """
    seconds_per_day = 86400
    seconds_in_month = days_in_month * seconds_per_day

    precip_rate_mm_mth = rate * seconds_in_month * 1000
    return precip_rate_mm_mth

def kelvin_to_celsius(temperature: xr.DataArray) -> xr.DataArray:
    """
    Convert temperature from Kelvin to Celsius.
    """
    return temperature - 273.15

if __name__ == "__main__":
    data_directory = "../data/Model_Output_From_Harrison/Temp_Precip/nw_ur_150_07"
    file_pattern = "nw_ur_150_07.cam.h0.*.nc"
    gadm_path = "../data/GADM/gadm_410.gpkg"
    dataset = load_climate_files(data_directory, file_pattern)
    days_in_month = calculate_days_in_month(dataset)
    dataset["days_in_month"] = days_in_month
    total_precip_rate = calculate_total_precipitation_rate(dataset)
    total_precip_rate_mm_mth = convert_precip_to_mm_month(total_precip_rate, dataset["days_in_month"])
    dataset["precip_mm_month"] = total_precip_rate_mm_mth
    dataset["precip_mm_month"].attrs = {
    "units": "mm month-1",
    "description": "Total monthly precipitation derived from PRECC and PRECL"
    }
    dataset["t_mean_celsius"] = kelvin_to_celsius(dataset.TS)
    dataset["t_mean_celsius"].attrs = {
        "units": "Celsius",
        "description": "Surface temperature converted from Kelvin to Celsius"
    }
    land_mask = create_land_mask(dataset, gadm_path)
    save_land_mask(land_mask, "../data/interim/land_mask/land_mask_nw_ur_150_07.nc")
    dataset_masked = dataset.where(land_mask)
    dataset_masked = dataset_masked.drop_vars('time_bnds')
    dataset_masked.to_netcdf("../data/interim/preprocessed/climate_data_processed.nc")