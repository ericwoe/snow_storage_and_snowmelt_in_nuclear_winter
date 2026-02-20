import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cftime
from basin_analysis import create_mask
import geopandas as gpd


def change_time(ds):
    # Neue Zeitwerte berechnen
    new_time = [
        cftime.DatetimeNoLeap(
            t.year if t.month > 1 else t.year - 1,
            t.month - 1 if t.month > 1 else 12,
            1,
        )
        for t in ds.time.values
    ]

    # Koordinate direkt ersetzen (in-place)
    ds.coords["time"] = new_time


def compute_cell_area(da):
    """
    Berechnet die Zellfläche (m²) für ein reguläres Lat-Lon-Gitter.

    Parameter
    ----------
    da : xarray.DataArray
        Muss lat- und lon-Koordinaten besitzen.

    Returns
    -------
    cell_area : xarray.DataArray
        2D DataArray (lat, lon) mit Zellflächen in m².
    """

    R = 6371000  # Erdradius in Metern

    # Gitterabstände (angenommen regelmäßig)
    dlat = np.deg2rad(abs(da.lat[1] - da.lat[0]))
    dlon = np.deg2rad(abs(da.lon[1] - da.lon[0]))

    # Breiten in Radiant
    lat_rad = np.deg2rad(da.lat)

    # Fläche pro Breitenband
    area_1d = R**2 * dlat * dlon * np.cos(lat_rad)

    # Zu 2D expandieren
    cell_area = area_1d.broadcast_like(da.isel(time=0))

    cell_area.name = "cell_area"
    cell_area.attrs["units"] = "m2"

    return cell_area


def sum_per_month(da, cell_area, mask):
    da_m = da / 1000
    volume = da_m * cell_area * mask
    volume_sum_per_month = volume.sum(dim=("lat", "lon"))
    return volume_sum_per_month


def mean_per_year(da):
    return da.groupby("time.year").mean(dim="time")


def annual_sum(da):
    return da.groupby("time.year").sum(dim="time")


def snow_covered_area_proportion_monthly(da, cell_area, mask):
    # Schnee vorhanden?
    snow = da > 0

    # Schneebedeckte Landfläche pro Monat
    snow_area = (snow * cell_area * mask).sum(dim=("lat", "lon"))

    # Gesamte Landfläche (konstant über Zeit)
    total_land_area = (cell_area * mask).sum(dim=("lat", "lon"))

    # Prozentuale Bedeckung
    percentage = (snow_area / total_land_area) * 100

    return percentage


def plot_global_snow_sum_per_month(*datasets, cell_area=None, mask=None, variable=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        snow_sum_monthly = sum_per_month(ds[variable], cell_area=cell_area, mask=mask)
        snow_sum_monthly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_sum_per_year(*datasets, cell_area=None, mask=None, variable=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        mean_snow_sum = mean_per_year(
            sum_per_month(ds[variable], cell_area=cell_area, mask=mask)
        )
        mean_snow_sum.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_sum_anomaly_per_year(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        mean_snow_sum_anomaly = mean_per_year(
            sum_per_month(
                ds[variable] - control[variable], cell_area=cell_area, mask=mask
            )
        )
        mean_snow_sum_anomaly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_sum_anomaly_per_month(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        monthly_snow_sum_anomaly = sum_per_month(
            ds[variable] - control[variable], cell_area=cell_area, mask=mask
        )
        monthly_snow_sum_anomaly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_cover_monthly(*datasets, cell_area=None, mask=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        snow_cover_monthly = snow_covered_area_proportion_monthly(
            ds.snow_storage, cell_area=cell_area, mask=mask
        )

        snow_cover_monthly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_cover_mean_per_year(*datasets, cell_area=None, mask=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        snow_cover_year = mean_per_year(
            snow_covered_area_proportion_monthly(
                ds.snow_storage, cell_area=cell_area, mask=mask
            )
        )

        snow_cover_year.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_cover_anomaly_monthly(
    *datasets, control=None, cell_area=None, mask=None
):

    fig, ax = plt.subplots(figsize=(8, 5))
    snow_cover_control = snow_covered_area_proportion_monthly(
        control.snow_storage, cell_area=cell_area, mask=mask
    )
    for i, ds in enumerate(datasets):
        snow_cover_monthly = snow_covered_area_proportion_monthly(
            ds.snow_storage, cell_area=cell_area, mask=mask
        )

        (snow_cover_monthly - snow_cover_control).plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_cover_anomaly_per_year(
    *datasets, control=None, cell_area=None, mask=None
):

    fig, ax = plt.subplots(figsize=(8, 5))
    snow_cover_control_mean_year = mean_per_year(
        snow_covered_area_proportion_monthly(
            control.snow_storage, cell_area=cell_area, mask=mask
        )
    )
    for i, ds in enumerate(datasets):
        snow_cover_mean_year = mean_per_year(
            snow_covered_area_proportion_monthly(
                ds.snow_storage, cell_area=cell_area, mask=mask
            )
        )

        (
            (snow_cover_mean_year - snow_cover_control_mean_year)
            / snow_cover_control_mean_year
            * 100
        ).plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_melt_per_month(
    *datasets, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        snow_sum_monthly = sum_per_month(ds[variable], cell_area=cell_area, mask=mask)
        snow_sum_monthly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_melt_per_year(*datasets, cell_area=None, mask=None, variable=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        mean_snow_sum = annual_sum(
            sum_per_month(ds[variable], cell_area=cell_area, mask=mask)
        )
        mean_snow_sum.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_melt_anomaly_per_year(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        mean_snow_sum_anomaly = annual_sum(
            sum_per_month(
                ds[variable] - control[variable], cell_area=cell_area, mask=mask
            )
        )
        mean_snow_sum_anomaly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ds_t5 = xr.open_dataset("./results/snow_57.nc")
    ds_t4 = xr.open_dataset("./results/snow_47.nc")
    ds_150 = xr.open_dataset("./results/snow_150.nc")
    ds_ctrl = xr.open_dataset("./results/snow_control.nc")
    mask = xr.open_dataarray("./data/interim/land_mask_neu.nc")

    cell_area = compute_cell_area(ds_ctrl.snow_storage)

    for ds in [ds_t5, ds_t4, ds_150, ds_ctrl]:
        change_time(ds)

    plot_global_snow_sum_per_month(
        ds_t5,
        ds_t4,
        ds_150,
        ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_sum_per_year(
        ds_t5,
        ds_t4,
        ds_150,
        ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_sum_anomaly_per_month(
        ds_t5,
        ds_t4,
        ds_150,
        control=ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_sum_anomaly_per_year(
        ds_t5,
        ds_t4,
        ds_150,
        control=ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_cover_monthly(
        ds_t5, ds_t4, ds_150, ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_global_snow_cover_mean_per_year(
        ds_t5, ds_t4, ds_150, ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_global_snow_cover_anomaly_monthly(
        ds_t5, ds_t4, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_global_snow_cover_anomaly_per_year(
        ds_t5, ds_t4, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )

    main_rivers = {
        "Rhine": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040023010,
        ],
        "Yellow River": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040007850,
        ],
        "Nile": [
            "./data/HydroBasins/hybas_af_lev01-12_v1c/hybas_af_lev04_v1c.shp",
            1040034260,
        ],
        "Mississippi": [
            "./data/HydroBasins/hybas_na_lev01-12_v1c/hybas_na_lev04_v1c.shp",
            7040047060,
        ],
    }

    for river, (filepath, river_id) in main_rivers.items():
        # Select river from Data
        basins_lev4 = gpd.read_file(filepath)
        river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
        river_basin_diss = river_basin.dissolve()
        # Fraction of Raster Cells in River Basin
        river_mask = create_mask(ds_150, river_basin_diss)
        plot_global_snow_melt_per_month(
            ds_t5,
            ds_t4,
            ds_150,
            ds_ctrl,
            cell_area=cell_area,
            mask=river_mask,
            variable="snow_melt",
        )
        plot_global_snow_melt_per_year(
            ds_t5,
            ds_t4,
            ds_150,
            ds_ctrl,
            cell_area=cell_area,
            mask=river_mask,
            variable="snow_melt",
        )
        plot_global_snow_melt_anomaly_per_year(
            ds_t5,
            ds_t4,
            ds_150,
            control=ds_ctrl,
            cell_area=cell_area,
            mask=river_mask,
            variable="snow_melt",
        )
