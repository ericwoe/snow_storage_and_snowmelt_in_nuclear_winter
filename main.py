from src.preprocessing import prepare_data
from src.preprocessing.land_mask import create_mask
from src.processing.snow_model import add_snow_variables
import xarray as xr
from pathlib import Path
import geopandas as gpd


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


datasets = {}

# Read in Data
for name, config in SCENARIOS.items():
    ds = prepare_data.run_preprocessing(
        data_directory=config["data_dir"],
        file_pattern=config["pattern"],
        output_path=None,
    )
    datasets[name] = ds

# Align all Datasets from Dictionary "datasets"
# along all 3 dimensions (time, lat, lon)
aligned = xr.align(*datasets.values(), join="inner")

# Back to Dictionary
datasets = dict(zip(datasets.keys(), aligned))

for name, ds in datasets.items():

    ds = add_snow_variables(ds)

    # Run Snow Model
    ds = add_snow_variables(ds)

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

    # Apply land mask to dataset
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
