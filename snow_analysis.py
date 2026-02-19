import xarray as xr
import numpy as np
from src.processing import snow_model
import matplotlib.pyplot as plt
import cftime


def change_time(ds):
    # Zeitkoordinate um einen Monat verschieben
    ds_new_time = ds.assign_coords(
        time=[
            cftime.DatetimeNoLeap(
                t.year if t.month > 1 else t.year - 1,
                t.month - 1 if t.month > 1 else 12,
                1,
            )
            for t in ds.time.values
        ]
    )
    return ds_new_time


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


def percentage_snow_cover_monthly(da, cell_area, mask):
    # Schnee vorhanden?
    snow = da > 0

    # Schneebedeckte Landfläche pro Monat
    snow_area = (snow * cell_area * mask).sum(dim=("lat", "lon"))

    # Gesamte Landfläche (konstant über Zeit)
    total_land_area = (cell_area * mask).sum(dim=("lat", "lon"))

    # Prozentuale Bedeckung
    percentage = (snow_area / total_land_area) * 100

    return percentage


def plot_global_snow_sum_per_year(*arrays, cell_area=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, da in enumerate(arrays):
        snow_sum = global_snow_sum_per_year(da.snow_storage, cell_area)
        snow_sum.plot(ax=ax, label=da.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_global_snow_sum_anomaly_per_year(*arrays, control=None, cell_area=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, da in enumerate(arrays):
        snow_sum = global_snow_sum_per_year(
            da.snow_storage - control.snow_storage, cell_area
        )
        snow_sum.plot(ax=ax, label=da.case)

    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    ds_t5 = xr.open_dataset("./results/snow_dataset_nw_targets_05.nc")
    ds_t4 = xr.open_dataset("./results/snow_dataset_nw_targets_04.nc")
    ds_150 = xr.open_dataset("./results/snow_dataset_150_tg.nc")
    ds_ctrl = xr.open_dataset("./results/snow_dataset_nw_cntrl_03.nc")

    cell_area = compute_cell_area(ds_ctrl.snow_storage)

    # Align the Datasets (time dimension)
    ds_ctrl_al, ds_t4_al, ds_t5_al, ds_150_al = xr.align(
        ds_ctrl, ds_t4, ds_t5, ds_150, join="inner"
    )

    # recalculate snow_storage and snow_melt for control scenario
    # snow_storage und snow_melt löschen
    ds_ctrl_al = ds_ctrl_al.drop_vars(["snow_storage", "snow_melt"])

    # snow model neu rechnen
    snow_model.add_snow_variables(ds_ctrl_al)

    all_ds = [ds_ctrl_al, ds_t4_al, ds_t5_al, ds_150_al]
    nw_sc = [ds_t4_al, ds_t5_al, ds_150_al]
    control = ds_ctrl_al
    plot_global_snow_sum_per_year(*all_ds, cell_area=cell_area)
    plot_global_snow_sum_anomaly_per_year(*nw_sc, control=control, cell_area=cell_area)
