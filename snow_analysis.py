import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cftime
from src.preprocessing.land_mask import create_mask
import geopandas as gpd
import os
from matplotlib.colors import TwoSlopeNorm, SymLogNorm
import matplotlib.colors as mcolors


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


def sum_per_month(
    da: xr.DataArray, cell_area: xr.DataArray, mask: xr.DataArray
) -> xr.DataArray:
    """
    Computes the total global snow volume per month.

    Converts snow depth from millimeters to meters, multiplies by grid cell area
    and a land mask to obtain the snow volume per cell, then sums spatially over
    all grid cells.

    Parameters
    ----------
    da : xr.DataArray
        Snow depth in millimeters, with dimensions (time, lat, lon).
    cell_area : xr.DataArray
        Area of each grid cell in square meters, with dimensions (lat, lon).
    mask : xr.DataArray
        Binary land mask (1 = land, 0 = ocean/exclude), with dimensions (lat, lon).

    Returns
    -------
    xr.DataArray
        Total snow volume in cubic meters per month, with dimension (time,).

    Example
    -------
    >>> snow_volume = sum_per_month(ds.snow_depth, cell_area, land_mask)
    """
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


def compute_zonal_snow_cover(da, mask):
    """
    Berechnet den Anteil der schneebedeckten Landfläche pro Breitengrad und Zeitschritt,
    gewichtet mit dem Landanteil (mask).
    Parameter
    ----------
    da : xarray.DataArray
        snow_storage mit Dimensionen (time, lat, lon)
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1), Dimensionen (lat, lon)
    Returns
    -------
    zonal_snow_cover : xarray.DataArray
        Anteil der schneebedeckten Landfläche in % (time, lat)
    """
    snow = (da > 0).astype(float)
    snow_land = (snow * mask).sum(dim="lon")
    total_land = mask.sum(dim="lon")
    zonal_snow_cover = (snow_land / total_land.where(total_land > 0)) * 100
    return zonal_snow_cover


def compute_zonal_snow_cover_anomaly(da, da_control, mask):
    """
    Berechnet die Anomalie der zonalen Schneebedeckung
    gegenüber dem Control-Szenario.
    Parameter
    ----------
    da : xarray.DataArray
        snow_storage des Szenarios (time, lat, lon)
    da_control : xarray.DataArray
        snow_storage des Control-Szenarios (time, lat, lon)
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1), Dimensionen (lat, lon)
    Returns
    -------
    zonal_anomaly : xarray.DataArray
        Zonale Anomalie der Schneebedeckung in Prozentpunkten (time, lat)
    """
    zonal_scenario = compute_zonal_snow_cover(da, mask)
    zonal_control = compute_zonal_snow_cover(da_control, mask)
    return zonal_scenario - zonal_control


def plot_hovmoeller_snow_cover_anomaly(*datasets, control=None, mask=None, titles=None):

    n = len(datasets)
    all_anomalies = []
    for ds in datasets:
        anomaly = compute_zonal_snow_cover_anomaly(
            ds.snow_storage, control.snow_storage, mask
        )
        all_anomalies.append(anomaly)

    all_values = np.concatenate([a.values.flatten() for a in all_anomalies])
    all_values = all_values[~np.isnan(all_values)]
    vabs = np.percentile(np.abs(all_values), 99)
    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    fig, axes = plt.subplots(
        n, 1, figsize=(14, 3 * n), sharex=True, sharey=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    for i, (ax, ds, anomaly) in enumerate(zip(axes, datasets, all_anomalies)):
        time_vals = np.arange(len(ds.time))
        lat_vals = ds.lat.values
        im = ax.pcolormesh(
            time_vals,
            lat_vals,
            anomaly.values.T,
            cmap="RdBu",
            norm=norm,
            shading="nearest",
        )
        ax.set_ylabel("Latitude [°]")

        # Dynamischer Titel
        label = titles[i] if titles and i < len(titles) else f"Scenario {i+1}"
        ax.set_title(f"Anomaly of zonal snow cover – {label}", fontsize=11)

        year_ticks = np.arange(0, len(ds.time), 12)
        year_labels = [ds.time.values[i].year for i in year_ticks]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

    axes[-1].set_xlabel("Year")
    fig.colorbar(
        im,
        ax=axes,
        label="Δ Snow covered land area [%]",
        location="right",
        shrink=0.6,
        pad=0.015,
        aspect=30,
    )
    fig.savefig(
        "./results/intercomparison/hovmoeller/snow_cover_anomaly_percent.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_hovmoeller_single(ds, mask=None):
    """
    Erstellt ein einzelnes Hovmöller-Diagramm (Zeit x Breitengrad) der zonalen
    Schneebedeckung.

    Parameter
    ----------
    ds : xarray.Dataset
        Dataset mit snow_storage und Attribut 'case'
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1)
    """
    zonal = compute_zonal_snow_cover(ds.snow_storage, mask)

    time_vals = np.arange(len(ds.time))
    lat_vals = ds.lat.values

    fig, ax = plt.subplots(figsize=(12, 5))

    im = ax.pcolormesh(
        time_vals,
        lat_vals,
        zonal.values.T,
        cmap="Blues",
        vmin=0,
        vmax=100,
        shading="nearest",
    )

    year_ticks = np.arange(0, len(ds.time), 12)
    year_labels = [ds.time.values[i].year for i in year_ticks]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(year_labels)

    ax.set_xlabel("Year")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"Zonal Snow Cover – {ds.case}")

    fig.colorbar(im, ax=ax, label="Snow covered land area [%]", shrink=0.8)

    plt.tight_layout()
    plt.show()


def compute_zonal_mean_snow_storage(da, mask):
    """
    Berechnet den zonal gemittelten snow_storage pro Breitengrad und Zeitschritt,
    gewichtet mit dem Landanteil (mask).

    Parameter
    ----------
    da : xarray.DataArray
        snow_storage mit Dimensionen (time, lat, lon)
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1), Dimensionen (lat, lon)

    Returns
    -------
    zonal_mean : xarray.DataArray
        Zonal gemittelter snow_storage (time, lat)
    """
    weighted_sum = (da * mask).sum(dim="lon")
    total_land = mask.sum(dim="lon")
    zonal_mean = weighted_sum / total_land.where(total_land > 0)
    return zonal_mean


def plot_hovmoeller_mean_snow_storage(
    *datasets, mask=None, savedir="./results/allgemeine_muster"
):
    """
    Erstellt Hovmöller-Diagramme des zonal gemittelten snow_storage
    für mehrere Szenarien in einem Multi-Panel-Plot.

    Parameter
    ----------
    *datasets : xarray.Dataset
        Beliebig viele Datasets mit snow_storage und Attribut 'case'
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1)
    savedir : str
        Verzeichnis zum Speichern der Abbildung
    """
    from matplotlib.colors import LogNorm

    os.makedirs(savedir, exist_ok=True)

    n = len(datasets)

    # Zonal gemittelte Werte vorab berechnen
    all_zonal = []
    for ds in datasets:
        zonal = compute_zonal_mean_snow_storage(ds.snow_storage, mask)
        all_zonal.append(zonal)

    # Gemeinsames vmax aus 95. Perzentil
    all_values = np.concatenate([z.values.flatten() for z in all_zonal])
    all_values = all_values[~np.isnan(all_values)]
    all_values = all_values[all_values > 0]
    vmin = 1  # Untergrenze für LogNorm (0 geht nicht)
    vmax = np.percentile(all_values, 99)

    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True, sharey=True)

    if n == 1:
        axes = [axes]

    norm = LogNorm(vmin=vmin, vmax=vmax)

    for ax, ds, zonal in zip(axes, datasets, all_zonal):
        time_vals = np.arange(len(ds.time))
        lat_vals = ds.lat.values

        # Werte < vmin auf NaN setzen damit LogNorm nicht kracht
        plot_data = zonal.values.T.copy()
        plot_data[plot_data < vmin] = np.nan

        im = ax.pcolormesh(
            time_vals,
            lat_vals,
            plot_data,
            cmap="YlGnBu",
            norm=norm,
            shading="nearest",
        )

        ax.set_ylabel("Latitude [°]")
        ax.set_title(ds.case, fontsize=11)

        year_ticks = np.arange(0, len(ds.time), 12)
        year_labels = [ds.time.values[i].year for i in year_ticks]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

    axes[-1].set_xlabel("Year")

    fig.colorbar(
        im,
        ax=axes,
        label="Snow storage [mm]",
        location="right",
        shrink=0.6,
        pad=0.015,
        aspect=30,
    )

    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/hovmoeller/mean_snow_storage.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def compute_zonal_snow_storage_mean_anomaly_absolute(da, da_control, mask):
    """
    Berechnet die prozentuale Anomalie des zonal gemittelten snow_storage
    gegenüber dem Control-Szenario.

    Parameter
    ----------
    da : xarray.DataArray
        snow_storage des Szenarios (time, lat, lon)
    da_control : xarray.DataArray
        snow_storage des Control-Szenarios (time, lat, lon)
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1), Dimensionen (lat, lon)

    Returns
    -------
    zonal_anomaly_pct : xarray.DataArray
        Prozentuale zonale Anomalie des snow_storage (time, lat)
    """
    zonal_scenario = compute_zonal_mean_snow_storage(da, mask)
    zonal_control = compute_zonal_mean_snow_storage(da_control, mask)

    # Prozentuale Änderung, nur wo Control > 0
    zonal_anomaly_absolute = zonal_scenario - zonal_control

    return zonal_anomaly_absolute


def plot_hovmoeller_snow_storage_anomaly(
    *datasets, control=None, mask=None, titles=None
):
    """
    Erstellt Hovmöller-Diagramme der prozentualen Anomalie des zonal
    gemittelten snow_storage gegenüber dem Control-Szenario.
    Verwendet eine klassenbasierte Farbskala via BoundaryNorm.

    Parameter
    ----------
    *datasets : xarray.Dataset
        Szenarien (ohne Control)
    control : xarray.Dataset
        Control-Szenario
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1)
    """

    n = len(datasets)
    vmin = 0
    vmax = 0
    all_anomalies = []

    for ds in datasets:
        anomaly = compute_zonal_snow_storage_mean_anomaly_absolute(
            ds.snow_storage, control.snow_storage, mask
        )
        all_anomalies.append(anomaly)
        vmin = min(vmin, anomaly.min().item())
        vmax = max(vmax, anomaly.max().item())

    print(vmin, vmax)

    bin_edges = np.array(
        [
            vmin,
            -4000,
            -2000,
            -1000,
            -500,
            -100,
            -50,
            -10,
            10,
            50,
            100,
            500,
            1000,
            2000,
            3000,
            vmax,
        ]
    )

    n_classes = len(bin_edges) - 1
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=n_classes)
    cmap = plt.get_cmap("RdBu", n_classes)

    fig, axes = plt.subplots(
        n, 1, figsize=(14, 3 * n), sharex=True, sharey=True, constrained_layout=True
    )

    if n == 1:
        axes = [axes]

    for i, (ax, ds, anomaly) in enumerate(zip(axes, datasets, all_anomalies)):
        time_vals = np.arange(len(ds.time))
        lat_vals = ds.lat.values

        plot_data = anomaly.values.T.copy()
        plot_data = np.nan_to_num(anomaly.values.T, nan=0.0)

        im = ax.pcolormesh(
            time_vals,
            lat_vals,
            plot_data,
            cmap=cmap,
            norm=norm,
            shading="nearest",
        )

        ax.set_ylabel("Latitude [°]")
        label = titles[i] if titles and i < len(titles) else f"Scenario {i+1}"
        ax.set_title(f"Anomaly of zonal mean snow storage – {label}", fontsize=11)

        year_ticks = np.arange(0, len(ds.time), 12)
        year_labels = [ds.time.values[j].year for j in year_ticks]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

    axes[-1].set_xlabel("Year")

    cbar = fig.colorbar(
        im,
        ax=axes,
        label="Δ Snow storage [mm]",
        location="right",
        shrink=0.6,
        pad=0.015,
        aspect=30,
        ticks=bin_edges,
    )
    cbar.ax.set_yticklabels([f"{v:g}" for v in bin_edges])

    fig.savefig(
        "./results/intercomparison/hovmoeller/mean_snow_storage_anomaly_absolute.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def compute_zonal_snow_storage_sum(da, mask, cell_area):
    """
    Berechnet den zonal gemittelten snow_storage pro Breitengrad und Zeitschritt,
    gewichtet mit dem Landanteil (mask).

    Parameter
    ----------
    da : xarray.DataArray
        snow_storage mit Dimensionen (time, lat, lon)
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1), Dimensionen (lat, lon)

    Returns
    -------
    zonal_mean : xarray.DataArray
        Zonal gemittelter snow_storage (time, lat)
    """
    weighted_sum = (da * mask * cell_area).sum(dim="lon")
    return weighted_sum


def compute_zonal_snow_storage_sum_anomaly_absolute(da, da_control, mask, cell_area):
    """
    Berechnet die prozentuale Anomalie des zonal gemittelten snow_storage
    gegenüber dem Control-Szenario.

    Parameter
    ----------
    da : xarray.DataArray
        snow_storage des Szenarios (time, lat, lon)
    da_control : xarray.DataArray
        snow_storage des Control-Szenarios (time, lat, lon)

    Returns
    -------
    zonal_anomaly_pct : xarray.DataArray
        Prozentuale zonale Anomalie des snow_storage (time, lat)
    """
    zonal_scenario = compute_zonal_snow_storage_sum(da, mask, cell_area)
    zonal_control = compute_zonal_snow_storage_sum(da_control, mask, cell_area)

    # Prozentuale Änderung, nur wo Control > 0
    zonal_anomaly_pct = zonal_scenario - zonal_control

    return zonal_anomaly_pct


def plot_hovmoeller_snow_storage_sum_anomaly_absolute(
    *datasets, control=None, mask=None, cell_area=None, titles=None
):
    """
    Erstellt Hovmöller-Diagramme der absoluten Anomalie der zonalen
    Schneespeicher-Summe gegenüber dem Control-Szenario.
    Verwendet eine symmetrische SymLogNorm-Skala mit Weiß bei 0.

    Parameter
    ----------
    *datasets : xarray.Dataset
        Szenarien (ohne Control)
    control : xarray.Dataset
        Control-Szenario
    mask : xarray.DataArray
        Landanteil pro Zelle (0 bis 1)
    cell_area : xarray.DataArray
        Fläche einer Zelle in m²
    titles : list of str, optional
        Titel für die einzelnen Subplots
    """

    n = len(datasets)
    all_anomalies = []
    vmin, vmax = 0, 0

    for ds in datasets:
        anomaly = compute_zonal_snow_storage_sum_anomaly_absolute(
            ds.snow_storage, control.snow_storage, mask, cell_area
        )
        all_anomalies.append(anomaly)
        vmin = min(vmin, anomaly.min().item())
        vmax = max(vmax, anomaly.max().item())
    # Symmetrische Skala damit 0 = weiß
    vabs = max(abs(vmin), abs(vmax))

    # linthresh: linearer Bereich um 0, ca. 0.1% von vabs
    norm = SymLogNorm(linthresh=1, vmin=-vabs, vmax=vabs)

    fig, axes = plt.subplots(
        n, 1, figsize=(14, 3 * n), sharex=True, sharey=True, constrained_layout=True
    )
    if n == 1:
        axes = [axes]

    for i, (ax, ds, anomaly) in enumerate(zip(axes, datasets, all_anomalies)):
        time_vals = np.arange(len(ds.time))
        lat_vals = ds.lat.values

        # NaN durch 0 ersetzen damit pcolormesh nichts ausblendet
        plot_data = np.nan_to_num(anomaly.values.T, nan=0.0)

        im = ax.pcolormesh(
            time_vals,
            lat_vals,
            plot_data,
            cmap="RdBu",
            norm=norm,
            shading="nearest",
        )
        ax.set_ylabel("Latitude [°]")
        label = titles[i] if titles and i < len(titles) else f"Scenario {i+1}"
        ax.set_title(
            f"Absolute Anomaly of zonal snow storage sum – {label}", fontsize=11
        )
        year_ticks = np.arange(0, len(ds.time), 12)
        year_labels = [ds.time.values[j].year for j in year_ticks]
        ax.set_xticks(year_ticks)
        ax.set_xticklabels(year_labels)

    axes[-1].set_xlabel("Year")
    fig.colorbar(
        im,
        ax=axes,
        label="Δ Snow storage [m³]",
        location="right",
        shrink=0.6,
        pad=0.015,
        aspect=30,
    )
    fig.savefig(
        "./results/intercomparison/hovmoeller/mean_snow_storage_sum_anomaly_absolute.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_global_snow_sum_per_month(*datasets, cell_area=None, mask=None, variable=None):

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, ds in enumerate(datasets):
        snow_sum_monthly = sum_per_month(ds[variable], cell_area=cell_area, mask=mask)
        snow_sum_monthly.plot(ax=ax, label=ds.case)
    ax.set_title("Global Snow Storage Sum per Month", fontsize=11)
    ax.set_ylabel("Snow Storage Volume [m³]")
    ax.set_xlabel("Zeit")
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/global_snow_sum_per_month.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_sum_per_year.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_sum_anomaly_per_year.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_global_snow_sum_anomaly_per_month(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))
    print("Summe Anomalien")
    for i, ds in enumerate(datasets):
        monthly_snow_sum_ds = sum_per_month(
            ds[variable], cell_area=cell_area, mask=mask
        )
        monthly_snow_sum_control = sum_per_month(
            control[variable], cell_area=cell_area, mask=mask
        )
        anomaly = monthly_snow_sum_ds - monthly_snow_sum_control
        anomaly.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Snow Volume [m³]")
    ax.set_title("Monthly Anomaly of Global Snow Sum Relative to Control", fontsize=11)
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/global_snow_sum_anomaly_per_month.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_absolute_global_snow_sum_anomaly_per_month(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))
    print("Summe absoluter Betrag Anomalien")
    for i, ds in enumerate(datasets):
        monthly_snow_sum_anomaly = sum_per_month(
            np.abs(ds[variable] - control[variable]), cell_area=cell_area, mask=mask
        )
        monthly_snow_sum_anomaly.plot(ax=ax, label=ds.case)
        print(
            f'{ds.case} Zeitpunkt d. Maximums: {monthly_snow_sum_anomaly.idxmax(dim="time")}'
        )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Snow Volume [m³]")
    ax.set_title(
        "Betrag von Monthly Anomaly of Global Snow Sum Relative to Control", fontsize=11
    )
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/global_snow_sum_anomaly_per_month_absolute.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_global_snow_sum_only_positive_anomaly_per_month(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):

    fig, ax = plt.subplots(figsize=(8, 5))
    print("Summe nur positiver Anomalien")
    for i, ds in enumerate(datasets):
        anomaly = ds[variable] - control[variable]
        monthly_snow_sum_anomaly = sum_per_month(
            anomaly.where(anomaly > 0), cell_area=cell_area, mask=mask
        )
        monthly_snow_sum_anomaly.plot(ax=ax, label=ds.case)
        print(
            f'{ds.case} Zeitpunkt d. Maximums: {monthly_snow_sum_anomaly.idxmax(dim="time")}'
        )

    ax.set_xlabel("Zeit")
    ax.set_ylabel("Snow Volume [m³]")
    ax.set_title("Summe der positivenAnomalien ", fontsize=11)
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/global_snow_sum_anomaly_only_positive.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_global_snow_sum_anomaly_per_month_percent(
    *datasets, control=None, cell_area=None, mask=None, variable=None
):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Globale Summe für Control berechnen (1D: nur time)

    for i, ds in enumerate(datasets):
        # Globale Summe für das jeweilige Dataset berechnen (1D: nur time)
        monthly_snow_sum_scenario = sum_per_month(
            ds[variable], cell_area=cell_area, mask=mask
        )
        print(monthly_snow_sum_scenario, len(monthly_snow_sum_scenario))
        monthly_snow_sum_control = sum_per_month(
            control[variable], cell_area=cell_area, mask=mask
        )
        print(monthly_snow_sum_control, len(monthly_snow_sum_control))

        monthly_anomaly_percent = (
            (monthly_snow_sum_scenario - monthly_snow_sum_control)
            / monthly_snow_sum_control
            * 100
        )

        monthly_anomaly_percent.plot(ax=ax, label=ds.case)

    ax.set_xlabel("Time")
    ax.set_ylabel("Snow Volume Anomaly [%]")
    ax.set_title("Monthly Anomaly of Global Snow Sum Relative to Control", fontsize=11)
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Nulllinie
    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/global_snow_sum_anomaly_per_month_percent.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_cover_per_month.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_cover_per_year.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_cover_anomaly_per_month.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_cover_anomaly_per_year.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_melt_per_month.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_melt_per_year.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


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
    fig.savefig(
        "./results/intercomparison/global_snow_melt_anomaly_per_year.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":

    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_ctrl = xr.open_dataset("./results/Control/snow_control.nc")
    mask = xr.open_dataarray("./data/interim/land_mask_neu.nc")

    cell_area = compute_cell_area(ds_ctrl.snow_storage)

    for ds in [ds_16, ds_47, ds_150, ds_ctrl]:
        change_time(ds)

    plot_global_snow_sum_per_month(
        ds_47,
        ds_16,
        ds_150,
        ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    """plot_global_snow_sum_per_month(
        ds_47.isel(time=slice(0, 7)),
        ds_16.isel(time=slice(0, 7)),
        ds_150.isel(time=slice(0, 7)),
        ds_ctrl.isel(time=slice(0, 7)),
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )"""

    plot_global_snow_sum_per_year(
        ds_47,
        ds_16,
        ds_150,
        ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_sum_anomaly_per_month(
        ds_47,
        ds_16,
        ds_150,
        control=ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_sum_only_positive_anomaly_per_month(
        ds_47,
        ds_16,
        ds_150,
        control=ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )

    plot_absolute_global_snow_sum_anomaly_per_month(
        ds_47,
        ds_16,
        ds_150,
        control=ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )

    plot_global_snow_sum_anomaly_per_year(
        ds_47,
        ds_16,
        ds_150,
        control=ds_ctrl,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
    )
    plot_global_snow_cover_monthly(
        ds_47, ds_16, ds_150, ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_global_snow_cover_mean_per_year(
        ds_47, ds_16, ds_150, ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_global_snow_cover_anomaly_monthly(
        ds_47, ds_16, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_global_snow_cover_anomaly_per_year(
        ds_47, ds_16, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
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
    plot_hovmoeller_snow_cover_anomaly(
        ds_16,
        ds_47,
        ds_150,
        control=ds_ctrl,
        mask=mask,
        titles=[
            "16 Tg",
            "47 Tg",
            "150 Tg",
        ],
    )
    plot_hovmoeller_snow_storage_anomaly(
        ds_16,
        ds_47,
        ds_150,
        control=ds_ctrl,
        mask=mask,
        titles=[
            "16 Tg",
            "47 Tg",
            "150 Tg",
        ],
    )
    plot_hovmoeller_snow_storage_sum_anomaly_absolute(
        ds_16,
        ds_47,
        ds_150,
        control=ds_ctrl,
        mask=mask,
        cell_area=cell_area,
        titles=[
            "16 Tg",
            "47 Tg",
            "150 Tg",
        ],
    )

    for river, (filepath, river_id) in main_rivers.items():
        # Select river from Data
        basins_lev4 = gpd.read_file(filepath)
        river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
        river_basin_diss = river_basin.dissolve()
        # Fraction of Raster Cells in River Basin
        river_mask = create_mask(ds_150, river_basin_diss)
        plot_global_snow_melt_per_month(
            ds_47,
            ds_16,
            ds_150,
            ds_ctrl,
            cell_area=cell_area,
            mask=river_mask,
            variable="snow_melt",
        )
        plot_global_snow_melt_per_year(
            ds_47,
            ds_16,
            ds_150,
            ds_ctrl,
            cell_area=cell_area,
            mask=river_mask,
            variable="snow_melt",
        )
        plot_global_snow_melt_anomaly_per_year(
            ds_47,
            ds_16,
            ds_150,
            control=ds_ctrl,
            cell_area=cell_area,
            mask=river_mask,
            variable="snow_melt",
        )
