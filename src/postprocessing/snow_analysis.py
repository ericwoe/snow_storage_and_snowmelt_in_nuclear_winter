import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cftime


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


def compute_grid_cell_area(da):
    """
    Berechnet die Fläche jeder Rasterzelle in km².
    Gibt ein 1D-DataArray (nur Latitude) zurück – xarray broadcastet automatisch.
    """
    R = 6371000.0  # Erdradius in m
    dlat = abs(da.lat[1] - da.lat[0])
    dlon = abs(da.lon[1] - da.lon[0])
    lat_rad = np.deg2rad(da.lat)
    dlat_rad = np.deg2rad(dlat)
    dlon_rad = np.deg2rad(dlon)

    lat_upper = np.clip(lat_rad + dlat_rad / 2, -np.pi / 2, np.pi / 2)
    lat_lower = np.clip(lat_rad - dlat_rad / 2, -np.pi / 2, np.pi / 2)

    area_1d = R**2 * dlon_rad * np.abs(np.sin(lat_upper) - np.sin(lat_lower))

    # Zu 2D expandieren
    cell_area = area_1d.broadcast_like(da.isel(time=0))
    cell_area.name = "cell_area"
    cell_area.attrs["units"] = "m2"
    return cell_area


def plot_combined_snow_analysis(
    *datasets,
    control=None,
    cell_area=None,
    mask=None,
    variable=None,
    labels=[16, 47, 150],
    colors=None,
):
    from matplotlib.ticker import MultipleLocator

    plt.style.use(
        "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)

    # Standardfarben falls keine übergeben
    if colors is None:
        colors = cmap_3_colors

    weights = cell_area * mask

    mean_snow_control = (control[variable] * weights).sum(
        dim=["lat", "lon"]
    ) / weights.sum(dim=["lat", "lon"])

    mean_snow_control_monthly = mean_snow_control.groupby("time.month").mean()
    std_snow_control_monthly = mean_snow_control.groupby("time.month").std()

    # snow cover control
    snow_control = control[variable] > 0

    # Schneebedeckte Landfläche pro Monat
    snow_area_control = (snow_control * weights).sum(dim=("lat", "lon"))

    # Gesamte Landfläche (konstant über Zeit)
    total_land_area_control = (weights).sum(dim=("lat", "lon"))

    # Prozentuale Bedeckung
    percentage_control = (snow_area_control / total_land_area_control) * 100
    # mittlere monatliche Bedeckung für Kontrollszenario (für Anomalien)
    mean_snow_cover_control_monthly = percentage_control.groupby("time.month").mean()
    std_snow_cover_control_monthly = percentage_control.groupby("time.month").std()

    n_months = len(datasets[0][variable].time)
    print(n_months)
    x = np.arange(n_months)

    for ds, label, color in zip(datasets, labels, colors):
        da = ds[variable]

        mean_snow_scenario = (da * weights).sum(dim=["lat", "lon"]) / weights.sum(
            dim=["lat", "lon"]
        )
        # 1. Plot
        axes[0].plot(x, mean_snow_scenario.values, label=f"{label} Tg BC", color=color)
        print(
            mean_snow_scenario.max(keep_attrs=True),
            f'{ds.case} Zeitpunkt d. Maximums: {mean_snow_scenario.idxmax(dim="time")}',
        )

        # 2. Plot
        anomaly_mean = (
            mean_snow_scenario.groupby("time.month") - mean_snow_control_monthly
        ).sortby("time")

        axes[1].plot(x, anomaly_mean.values, label=f"{label} Tg BC", color=color)

        print(
            anomaly_mean.max(keep_attrs=True),
            f'{ds.case} Zeitpunkt d. Maximums: {anomaly_mean.idxmax(dim="time")}',
        )

        snow = da > 0

        # Schneebedeckte Landfläche pro Monat
        snow_area = (snow * cell_area * mask).sum(dim=("lat", "lon"))

        # Gesamte Landfläche (konstant über Zeit)
        total_land_area = (cell_area * mask).sum(dim=("lat", "lon"))

        # Prozentuale Bedeckung
        percentage_scenario = (snow_area / total_land_area) * 100

        percentage_anomaly = percentage_scenario - percentage_control

        percentage_anomaly = (
            percentage_scenario.groupby("time.month") - mean_snow_cover_control_monthly
        ).sortby("time")

        print(
            percentage_anomaly.max(keep_attrs=True),
            f'{ds.case} Zeitpunkt d. Maximums: {percentage_anomaly.idxmax(dim="time")}',
        )

        # anomaly_pct = (
        #    (mean_snow - mean_snow_control) / mean_snow_control * 100
        # )
        axes[2].plot(x, percentage_anomaly.values, label=f"{label} Tg BC", color=color)

    axes[0].plot(
        x,
        mean_snow_control.values,
        label="Control",
        color="#E34444",
        linewidth=1.2,
        linestyle="--",
    )

    std_expanded = std_snow_control_monthly.sel(
        month=mean_snow_scenario.time.dt.month
    ).values

    axes[1].fill_between(
        x,
        -std_expanded,
        std_expanded,
        color="#CE5151",
        alpha=0.4,
        edgecolor="#CE5151",
        linewidth=1,
        label="±1σ Control",
    )

    std_expanded_cover = std_snow_cover_control_monthly.sel(
        month=anomaly_mean.time.dt.month
    ).values

    axes[2].fill_between(
        x,
        -std_expanded_cover,
        std_expanded_cover,
        color="#CE5151",
        alpha=0.4,
        edgecolor="#CE5151",
        linewidth=1,
        label="±1σ Control",
    )

    # --- X-Achse ---
    tick_positions = np.arange(0, n_months, 12)

    tick_labels = [f"Year {i}" for i in range(len(tick_positions))]

    for ax in axes:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlim(-1, n_months)
        ax.grid(which="major", axis="x", linewidth=0.8, color="gray", alpha=0.5)
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(which="minor", axis="x", linewidth=0.3, color="gray", alpha=0.3)
        ax.grid(which="major", axis="y", linewidth=0.5, color="gray", alpha=0.3)
        ax.legend(fontsize=9)

    axes[2].set_xticklabels(tick_labels, rotation=45, ha="right")
    axes[2].set_xlabel("Time [months]", fontsize=11)

    # axes[0].set_title("Monthly Mean Snow Storage", fontsize=12)
    axes[0].set_ylabel("Mean Snow Storage [mm]", fontsize=12)
    axes[0].axvline(4, color="black", linewidth=0.5, linestyle="dashed")

    # axes[1].set_title("Monthly Snow Storage Anomaly", fontsize=12)
    axes[1].set_ylabel("Mean Snow Storage Anomaly [mm]", fontsize=12)
    axes[1].axhline(0, color="#E34444", linewidth=0.8, linestyle="--")
    axes[1].axvline(4, color="black", linewidth=0.5, linestyle="dashed")

    # axes[2].set_title("Monthly Snow Cover Extent Anomaly", fontsize=12)
    axes[2].set_ylabel("Snow Cover Extent Anomaly [%-points]", fontsize=12)
    axes[2].axhline(0, color="#E34444", linewidth=1.2, linestyle="--")
    axes[2].axvline(4, color="black", linewidth=0.5, linestyle="dashed")

    for i, ax in enumerate(axes):
        ax.text(
            -0.05,
            1.02,
            f"{chr(97+i)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="right",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/global_analysis.png",
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

    cell_area = compute_grid_cell_area(ds_ctrl.snow_storage)

    colors = ["#abd9e9", "#74add1", "#4575b4"]

    # Ignore Cluster 0
    labels = np.load("./results/clustering/Control_scenario_dtw/3_cluster_labels.npy")
    template = ds_ctrl.snow_storage.isel(time=0)
    cluster_map = xr.full_like(template, fill_value=np.nan)
    land_mask = ~np.isnan(ds_ctrl.snow_storage.isel(time=0))
    cluster_map.values[land_mask.values] = labels

    for ds in [ds_16, ds_47, ds_150, ds_ctrl]:
        change_time(ds)

    ds_47_0 = ds_47.where(cluster_map != 0)
    ds_16_0 = ds_16.where(cluster_map != 0)
    ds_150_0 = ds_150.where(cluster_map != 0)
    ds_ctrl_0 = ds_ctrl.where(cluster_map != 0)

    plot_combined_snow_analysis(
        ds_16_0,
        ds_47_0,
        ds_150_0,
        control=ds_ctrl_0,
        cell_area=cell_area,
        mask=mask,
        variable="snow_storage",
        labels=[16, 47, 150],
        colors=colors,
    )
