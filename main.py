import src.preprocessing
import src.snow_model

DATA_DIR = "./data/Model_Output_From_Harrison/Temp_Precip/nw_ur_150_07"
PATTERN = "nw_ur_150_07.cam.h0.*.nc"

# Preprare Dataset
ds = src.preprocessing.run_preprocessing(
    data_directory=DATA_DIR, file_pattern=PATTERN, output_path=None
)

# Run Snow Model
ds = src.snow_model.add_snow_variables(ds)

print(ds.snow_storage)
