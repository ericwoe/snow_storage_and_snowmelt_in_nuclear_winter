import src.preprocessing
import src.snow_model
import src.land_mask
import xarray as xr
from pathlib import Path


DATA_DIR = "./data/Model_Output_From_Harrison/Temp_Precip/nw_ur_150_07"
PATTERN = "nw_ur_150_07.cam.h0.*.nc"
GADM_FILE_PATH = "./data/GADM/gadm_410.gpkg"
LAND_MASK_OUTPUT_PATH = "./data/interim/land_mask.nc"

# Preprare Dataset
ds = src.preprocessing.run_preprocessing(
    data_directory=DATA_DIR, file_pattern=PATTERN, output_path=None
)

# Run Snow Model
ds = src.snow_model.add_snow_variables(ds)


land_mask_path = Path(LAND_MASK_OUTPUT_PATH)
# Create or import land mask
if land_mask_path.exists():
    print("Loading existing land mask")
    mask = xr.open_dataarray(land_mask_path)
else:
    print("Creating land mask")
    mask = src.land_mask.create_land_mask(ds, gadm_path=GADM_FILE_PATH)
    print(f"Saving Mask to {LAND_MASK_OUTPUT_PATH}")
    # Ensure directory exists
    land_mask_path.parent.mkdir(parents=True, exist_ok=True)
    # Save mask to NetCDF
    mask.to_netcdf(land_mask_path)
