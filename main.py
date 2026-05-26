from src.preprocessing import prepare_data
from src.preprocessing.land_mask import create_mask
from src.processing.snow_model import add_snow_variables
from src.processing.clustering import (
    prepare_time_series,
    time_series_analysis,
    convert_labels_to_xarray,
)
import xarray as xr
from pathlib import Path
import geopandas as gpd
import cftime
import pickle
import numpy as np

SCENARIOS = {
    "Control": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_cntrl_03",
        "pattern": "nw_cntrl_03.cam.h0.*.nc",
        "result_path": "./results/Control/snow_control.nc",
    },
    "16": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_targets_04",
        "pattern": "nw_targets_04.cam.h0.*.nc",
        "result_path": "./results/16/snow_16.nc",
    },
    "47": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_targets_05",
        "pattern": "nw_targets_05.cam.h0.*.nc",
        "result_path": "./results/47/snow_47.nc",
    },
    "150": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_ur_150_07",
        "pattern": "nw_ur_150_07.cam.h0.*.nc",
        "result_path": "./results/150/snow_150.nc",
    },
}

GADM_FILE_PATH = "./data/ne_110m_land/ne_110m_land.shp"
LAND_MASK_FILE_PATH = "./data/interim/land_mask_neu.nc"
SPINUP_START = cftime.DatetimeNoLeap(1, 2, 1, 0, 0, 0, 0, has_year_zero=True)
SPINUP_END = cftime.DatetimeNoLeap(5, 1, 1, 0, 0, 0, 0, has_year_zero=True)
ANALYSIS_START = cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True)
ANALYSIS_END = cftime.DatetimeNoLeap(20, 1, 1, 0, 0, 0, 0, has_year_zero=True)


datasets = {}

# Read in Data and calculate precip and temperature variables
for name, config in SCENARIOS.items():
    ds = prepare_data.run_preprocessing(
        data_directory=config["data_dir"],
        file_pattern=config["pattern"],
        output_path=None,
    )
    print(ds)
    print(ds.time)
    datasets[name] = ds

# Add spinup period to all datasets
spinup_period = datasets["Control"].sel(time=slice(SPINUP_START, SPINUP_END))
for name, ds in datasets.items():
    if name == "Control":
        continue  # Control dataset already includes the spinup period
    datasets[name] = xr.concat([spinup_period, ds], dim="time")
    print(datasets[name].time)


# Align all Datasets from Dictionary "datasets"
# along all 3 dimensions (time, lat, lon)
# aligned = xr.align(*datasets.values(), join="inner")

# Back to Dictionary
# datasets = dict(zip(datasets.keys(), aligned))

for name, ds in datasets.items():

    # Run Snow Model
    ds = add_snow_variables(ds)

    # Reduce to Analysis Period
    ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))  # Select only analysis period
    print(ds)
    print(ds.time)

    # Create or import land mask
    land_mask_path = Path(LAND_MASK_FILE_PATH)
    if land_mask_path.exists():
        print("Loading existing land mask")
        mask = xr.open_dataarray(land_mask_path)
    else:
        print("Creating land mask")
        gadm = gpd.read_file(GADM_FILE_PATH)
        mask = create_mask(ds, gadm)
        print(f"Saving Mask to {LAND_MASK_FILE_PATH}")
        # Ensure directory exists
        land_mask_path.parent.mkdir(parents=True, exist_ok=True)
        # Save mask to NetCDF
        mask.to_netcdf(land_mask_path)

    # Apply land mask to dataset - values become NaN where mask == 0
    print("Applying land mask to dataset")
    ds = ds.where(mask > 0)

    # Delete time_bnds to avoid saving conflict
    print("Deleting time_bnds")
    ds = ds.drop_vars("time_bnds")

    # Save final dataset
    print(f"Saving {name} dataset to {SCENARIOS[name]['result_path']}")
    result_path = Path(SCENARIOS[name]["result_path"])
    # Ensure directory exists
    result_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(result_path)

    # Datasets Dict aktualisieren
    datasets[name] = ds
##################################################################################################
# CLUSTERING - 47 Tg Scenario and Control
##################################################################################################
# Prepare time series for clustering

timeseries = prepare_time_series(datasets["47"].snow_storage)

print(timeseries.shape)

# Subset for Elbow Method
# subset_size = int(0.2 * 5661)  # ca. 1248 Reihen
# indices = np.random.choice(5661, subset_size, replace=False)
# timeseries_subset = timeseries_scaled[indices]

labels, km = time_series_analysis(timeseries, n_clusters=5)
np.save(
    "./results/clustering/47_Tg_dtw/5_cluster_labels.npy", labels
)  # Labels als NumPy-Array
with open("./results/clustering/47_Tg_dtw/kmeans_model_5_clusters.pkl", "wb") as f:
    pickle.dump(km, f)  # Modell mit pickle

# Control Scenario Clustering
timeseries = prepare_time_series(datasets["Control"].snow_storage)
print(timeseries.shape)

labels, km = time_series_analysis(timeseries, n_clusters=5)
np.save(
    "./results/clustering/Control_scenario_dtw/3_cluster_labels.npy", labels
)  # Labels als NumPy-Array
with open(
    "./results/clustering/Control_scenario_dtw/kmeans_model_3_clusters.pkl", "wb"
) as f:
    pickle.dump(km, f)  # Modell mit pickle
