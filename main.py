import pickle
import tarfile
import urllib.request
from pathlib import Path

import cftime
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.preprocessing import prepare_data
from src.utilities import create_mask, compute_grid_cell_area
from src.processing.snow_model import add_snow_variables
from src.processing.clustering import (
    prepare_time_series,
    time_series_analysis,
    elbow_and_silhouette_method,
)
from src.postprocessing.snow_analysis import (
    change_time,
    plot_combined_snow_analysis,
    find_mean_anomaly_peak_time,
    plot_snow_anomaly_spatial,
    plot_hovmoeller_mean_snow_storage,
    DEFAULT_SCENARIO_COLORS,
)
from src.postprocessing.plot_clustering import plot_cluster_combined
from src.postprocessing.basin_analysis import (
    convert_mm_month_to_discharge_m3_month,
    monthly_discharge_sum_m3,
    plot_annual_anomaly,
    plot_monthly_bars_rain_snow,
)

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)

SCENARIOS = {
    "Control": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_cntrl_03",
        "pattern": "*.nc",
        "result_path": "./results/Control/snow_control.nc",
        "osf_url": "https://osf.io/download/w6qb5/",
        "osf_archive": "nw_cntrl_03.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
    "16": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_targets_04",
        "pattern": "*.nc",
        "result_path": "./results/16/snow_16.nc",
        "osf_url": "https://osf.io/download/n2vtj/",
        "osf_archive": "nw_targets_04.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
    "47": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_targets_05",
        "pattern": "*.nc",
        "result_path": "./results/47/snow_47.nc",
        "osf_url": "https://osf.io/download/4uysj/",
        "osf_archive": "nw_targets_05.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
    "150": {
        "data_dir": "./data/Model_Output_From_Harrison/Temp_Precip/nw_ur_150_07",
        "pattern": "*.nc",
        "result_path": "./results/150/snow_150.nc",
        "osf_url": "https://osf.io/download/wnvxf/",
        "osf_archive": "nw_ur_150_07.TS_TSMN_TSMX_PRECC_PRECL_v2024-12-12.nc.tar.gz",
    },
}

RESULT_DIRS = {
    name: Path(config["result_path"]).parent for name, config in SCENARIOS.items()
}

LAND_SHAPE_URL = "https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip"
LAND_MASK_FILE_PATH = "./data/interim/land_mask_neu.nc"
SPINUP_START = cftime.DatetimeNoLeap(1, 2, 1, 0, 0, 0, 0, has_year_zero=True)
SPINUP_END = cftime.DatetimeNoLeap(5, 1, 1, 0, 0, 0, 0, has_year_zero=True)
ANALYSIS_START = cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True)
ANALYSIS_END = cftime.DatetimeNoLeap(20, 1, 1, 0, 0, 0, 0, has_year_zero=True)


def _download_model_output(name, config):
    """Download and extract a scenario's model output from OSF if not already present."""
    data_dir = Path(config["data_dir"])
    if data_dir.exists() and any(data_dir.glob(config["pattern"])):
        print(f"  {name}: data already present, skipping download")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / config["osf_archive"]
    print(f"  {name}: downloading from OSF...")
    urllib.request.urlretrieve(config["osf_url"], archive_path)
    print(f"  {name}: extracting...")
    with tarfile.open(archive_path) as tar:
        for member in tar.getmembers():
            # The archive preserves the original server path; strip all directory
            # components so files land directly in data_dir.
            filename = Path(member.name).name
            if filename.endswith(".nc"):
                member.name = filename
                tar.extract(member, data_dir)
    archive_path.unlink()
    print(f"  {name}: done")


datasets = {}

# Read in Data and calculate precip and temperature variables
print("\n=== PREPROCESSING ===")
print("Checking model output data...")
for name, config in SCENARIOS.items():
    _download_model_output(name, config)
for name, config in SCENARIOS.items():
    ds = prepare_data.run_preprocessing(
        data_directory=config["data_dir"],
        file_pattern=config["pattern"],
        output_path=None,
    )
    datasets[name] = ds

# Extract spinup period from Control dataset
spinup_period = datasets["Control"].sel(time=slice(SPINUP_START, SPINUP_END))

# Create or import land mask (same grid for all scenarios)
land_mask_path = Path(LAND_MASK_FILE_PATH)
if land_mask_path.exists():
    print("Loading existing land mask")
    mask = xr.open_dataarray(land_mask_path)
else:
    print("Creating land mask")
    gadm = gpd.read_file(LAND_SHAPE_URL)
    mask = create_mask(datasets["Control"], gadm)
    print(f"Saving Mask to {LAND_MASK_FILE_PATH}")
    land_mask_path.parent.mkdir(parents=True, exist_ok=True)
    mask.to_netcdf(land_mask_path)


for name, config in SCENARIOS.items():
    ds = datasets[name]

    # Prepend spinup to non-Control datasets so the snow model initializes correctly
    if name != "Control":
        ds = xr.concat([spinup_period, ds], dim="time")

    # Run Snow Model on the full time series (spinup included) for correct initialization
    ds = add_snow_variables(ds)

    # Reduce to Analysis Period
    ds = ds.sel(time=slice(ANALYSIS_START, ANALYSIS_END))

    # Apply land mask - values become NaN where mask == 0
    print("Applying land mask to dataset")
    ds = ds.where(mask > 0)

    # Delete time_bnds to avoid saving conflict
    print("Deleting time_bnds")
    ds = ds.drop_vars("time_bnds")

    # Save preprocessed dataset
    print(f"Saving {name} dataset to {config['result_path']}")
    result_path = Path(config["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(result_path)

    datasets[name] = ds

cell_area = compute_grid_cell_area(datasets["Control"].snow_storage.isel(time=0))


##################################################################################################
# CLUSTERING - 47 Tg Scenario and Control
##################################################################################################

"""print("\n=== CLUSTERING ===")
for name, ds in datasets.items():
    if name not in ["47", "Control"]:
        continue

    clustering_dir = RESULT_DIRS[name] / "clustering"
    models_dir = clustering_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    timeseries = prepare_time_series(ds.snow_storage)

    # Subset for Elbow Method and Silhouette Score
    subset_size = int(0.3 * timeseries.shape[0])
    indices = np.random.choice(timeseries.shape[0], subset_size, replace=False)
    timeseries_subset = timeseries[indices]

    elbow_and_silhouette_method(
        timeseries_subset, max_clusters=10, save_dir=clustering_dir / "elbow_silhouette"
    )

    n_clusters = 5 if name == "47" else 3
    labels, km = time_series_analysis(timeseries, n_clusters=n_clusters)

    if name == "47":
        timeseries_control = prepare_time_series(datasets["Control"].snow_storage)

        land_mask = ~np.isnan(ds.snow_storage.isel(time=0))
        cell_area_1d = cell_area.values[land_mask.values]
        fractions_1d = mask.values[land_mask.values]

        timeseries_squeezed = timeseries.squeeze()
        timeseries_control_squeezed = timeseries_control.squeeze()

        plot_cluster_combined(
            da=ds.snow_storage,
            timeseries_scenario=timeseries_squeezed,
            timeseries_ctrl=timeseries_control_squeezed,
            labels=labels,
            cell_areas=cell_area_1d,
            fractions=fractions_1d,
            n_clusters=5,
            title=None,
            parameter_name="Snow Storage (mm)",
            save_path=clustering_dir,
        )

    # Save cluster labels and model
    np.save(models_dir / f"{n_clusters}_cluster_labels.npy", labels)
    with open(models_dir / f"kmeans_model_{n_clusters}_clusters.pkl", "wb") as f:
        pickle.dump(km, f)"""


##################################################################################################
# POSTPROCESSING - Global Snow Analysis
##################################################################################################

print("\n=== GLOBAL SNOW ANALYSIS ===")
# Shift time coordinates back by one month for all datasets
for ds in datasets.values():
    change_time(ds)

ds_47 = datasets["47"]
ds_control = datasets["Control"]

global_analysis_dir = RESULT_DIRS["47"] / "global_analysis"
global_analysis_dir.mkdir(parents=True, exist_ok=True)

# Exclude Cluster 0 (permanently accumulating cells) from analysis
# Load cluster labels for Control dataset
labels_ctrl = np.load(
    RESULT_DIRS["Control"] / "clustering" / "models" / "3_cluster_labels.npy"
)
# Map cluster labels to the spatial grid of the Control dataset
template = ds_control.snow_storage.isel(time=0)
cluster_map = xr.full_like(template, fill_value=np.nan)
land_mask_ctrl = ~np.isnan(ds_control.snow_storage.isel(time=0))
cluster_map.values[land_mask_ctrl.values] = labels_ctrl
# Exclude Cluster 0 from all datasets for analysis
ds_47_0 = ds_47.where(cluster_map != 0)
ds_16_0 = datasets["16"].where(cluster_map != 0)
ds_150_0 = datasets["150"].where(cluster_map != 0)
ds_ctrl_0 = ds_control.where(cluster_map != 0)

plot_combined_snow_analysis(
    ds_16_0,
    ds_47_0,
    ds_150_0,
    control=ds_ctrl_0,
    cell_area=cell_area,
    mask=mask,
    variable="snow_storage",
    labels=[16, 47, 150],
    colors=DEFAULT_SCENARIO_COLORS,
    output_path=global_analysis_dir / "global_analysis.png",
)

time_mean_peak = find_mean_anomaly_peak_time(
    scenario=ds_47_0,
    control=ds_ctrl_0,
    cell_area=cell_area,
    mask=mask,
    variable="snow_storage",
)

plot_snow_anomaly_spatial(
    scenario=ds_47,
    control=ds_control,
    time_point=time_mean_peak,
    scenario_name="47 Tg",
    variable="snow_storage",
    output_path=global_analysis_dir / "spatial_snow_anomaly.png",
    vmin=-5,
    vmax=5,
    linthresh=0.1,
)

plot_hovmoeller_mean_snow_storage(
    datasets["16"],
    ds_47,
    datasets["150"],
    control=ds_control,
    mask=mask,
    cell_area=cell_area,
    savedir=global_analysis_dir / "hovmoeller",
    titles=[16, 47, 150],
    vmin=-5,
    vmax=5,
    linthresh=0.1,
)


##################################################################################################
# POSTPROCESSING - Basin Analysis
##################################################################################################

print("\n=== BASIN ANALYSIS ===")

_HYDROBASINS_BASE = "https://data.hydrosheds.org/file/hydrobasins/standard"
HYDROBASINS_URLS = {
    "eu": f"{_HYDROBASINS_BASE}/hybas_eu_lev01-12_v1c.zip",
    "as": f"{_HYDROBASINS_BASE}/hybas_as_lev01-12_v1c.zip",
    "si": f"{_HYDROBASINS_BASE}/hybas_si_lev01-12_v1c.zip",
    "na": f"{_HYDROBASINS_BASE}/hybas_na_lev01-12_v1c.zip",
}
HYDROBASINS_CACHE_DIR = Path("./data/HydroBasins/cache")

MAIN_RIVERS = {
    "Rhine": ("eu", 2040023010),
    "Ganges-Brahmaputra-Meghna": ("as", 4040025450),
    "Yellow River": ("as", 4040007850),
    "Tigris-Euphrates": ("eu", 2040073570),
    "Ob": ("si", 3040001840),
    "Mississippi": ("na", 7040047060),
}


def _load_basin_region(region):
    cache_path = HYDROBASINS_CACHE_DIR / f"hybas_{region}_lev04.gpkg"
    if cache_path.exists():
        print(f"Loading HydroBasins region '{region}' from cache")
        return gpd.read_file(cache_path)
    HYDROBASINS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vsi_path = (
        f"/vsizip//vsicurl/{HYDROBASINS_URLS[region]}/hybas_{region}_lev04_v1c.shp"
    )
    print(f"Downloading HydroBasins region '{region}'")
    gdf = gpd.read_file(vsi_path)
    gdf.to_file(cache_path, driver="GPKG")
    return gdf


basin_analysis_dir = RESULT_DIRS["47"] / "basin_analysis"
basin_analysis_dir.mkdir(parents=True, exist_ok=True)

BASIN_VARIABLES = {
    "snowmelt": "snow_melt",
    "rain": "rain",
}
scenario_datasets = {"47": ds_47, "control": ds_control}

annual_snowmelt_anomaly_data = {}
annual_total_discharge_anomaly_data = {}
monthly_snow_rain_data = {}

# In-memory cache to avoid re-reading within one run (Rhine and Tigris share "eu",
# Ganges and Yellow River share "as")
_basin_gdf_cache = {}

for river, (region, river_id) in MAIN_RIVERS.items():
    if region not in _basin_gdf_cache:
        _basin_gdf_cache[region] = _load_basin_region(region)
    basins_lev4 = _basin_gdf_cache[region]
    river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
    river_basin_diss = river_basin.dissolve()
    river_mask = create_mask(ds_control, river_basin_diss)

    weights_basin = cell_area * river_mask

    dsc_annual = {}
    dsc_m3_s = {}
    for var_key, var_name in BASIN_VARIABLES.items():
        for scen_key, scen_ds in scenario_datasets.items():
            masked = scen_ds[var_name].where(river_mask > 0)
            dsc = convert_mm_month_to_discharge_m3_month(masked, river_mask, cell_area)
            dsc_sum = monthly_discharge_sum_m3(dsc)
            dsc_annual[scen_key, var_key] = dsc_sum.groupby("time.year").sum("time")
            dsc_m3_s[scen_key, var_key] = dsc_sum / (
                scen_ds["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
            )

    annual_total_discharge_anomaly_data[river] = (
        dsc_annual["47", "snowmelt"] + dsc_annual["47", "rain"],
        dsc_annual["control", "snowmelt"] + dsc_annual["control", "rain"],
    )

    annual_snowmelt_anomaly_data[river] = (
        dsc_annual["47", "snowmelt"],
        dsc_annual["control", "snowmelt"],
    )

    monthly_snow_rain_data[river] = {
        "scenario_snowmelt": dsc_m3_s["47", "snowmelt"],
        "scenario_rain": dsc_m3_s["47", "rain"],
        "control_snowmelt": dsc_m3_s["control", "snowmelt"],
        "control_rain": dsc_m3_s["control", "rain"],
    }

# --- Multi-panel figures ---

# Annual Anomalies of Total Discharge (Rain + Snowmelt)
n_rivers = len(annual_total_discharge_anomaly_data)
ncols = 3
nrows = (n_rivers + ncols - 1) // ncols

fig, axes = plt.subplots(
    nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True
)
axes_flat1 = axes.flatten()

for ax, (river, (scenario, control)), idx in zip(
    axes_flat1,
    annual_total_discharge_anomaly_data.items(),
    range(len(annual_total_discharge_anomaly_data)),
):
    plot_annual_anomaly(
        scenario, control, river=river, ax=ax, title="Annual Total Discharge"
    )
    ax.text(
        -0.10,
        1.07,
        f"{chr(97 + idx)})",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

y_max = max(ax.get_ylim()[1] for ax in axes_flat1[:n_rivers])
y_min = min(ax.get_ylim()[0] for ax in axes_flat1[:n_rivers])
for ax in axes_flat1[:n_rivers]:
    ax.set_ylim(y_min, y_max, auto=False)
for ax in axes_flat1[n_rivers:]:
    ax.set_visible(False)

plt.savefig(basin_analysis_dir / "river_basin_annual_anomalies.png", dpi=300)
plt.close(fig)


# Annual Anomalies of Snowmelt
fig4, axes4 = plt.subplots(
    nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True
)
axes_flat4 = axes4.flatten()

for ax, (river, (scenario, control)), idx in zip(
    axes_flat4,
    annual_snowmelt_anomaly_data.items(),
    range(len(annual_snowmelt_anomaly_data)),
):
    plot_annual_anomaly(scenario, control, river=river, ax=ax, title="Snowmelt")
    ax.text(
        -0.09,
        1.04,
        f"{chr(97 + idx)})",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

for ax in axes_flat4[n_rivers:]:
    ax.set_visible(False)

plt.savefig(basin_analysis_dir / "river_basin_annual_snowmelt_anomalies.png", dpi=300)
plt.close(fig4)

# Monthly Mean Discharge from Snowmelt and Rain
fig3, axes3 = plt.subplots(3, 2, figsize=(2 * 6, 3 * 5), constrained_layout=True)
axes_flat3 = axes3.flatten()

for ax, (river, river_data), idx in zip(
    axes_flat3, monthly_snow_rain_data.items(), range(len(monthly_snow_rain_data))
):
    plot_monthly_bars_rain_snow(
        river_data["scenario_snowmelt"],
        river_data["scenario_rain"],
        river_data["control_snowmelt"],
        river_data["control_rain"],
        river,
        ax=ax,
    )
    ax.text(
        -0.15,
        1.05,
        f"{chr(97 + idx)})",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        weight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
    )

for ax in axes_flat3[:n_rivers]:
    y_min, y_max = ax.get_ylim()
    y_abs_max = max(abs(y_min), abs(y_max))
    ax.set_ylim(-y_abs_max, y_abs_max)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])

plt.savefig(basin_analysis_dir / "all_rivers_monthly_bars_rain_snow.png", dpi=300)
plt.close(fig3)
