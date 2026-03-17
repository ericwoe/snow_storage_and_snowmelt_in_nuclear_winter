import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.processing.clustering import prepare_time_series
import xarray as xr
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import math


plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


def elbow_plot(inertias, save_path):
    """Plots the elbow method results
    Arguments:
        inertias: DataFrame with inertias for each k
        save_path: str - where to save the plot"""
    plt.figure()
    plt.plot(inertias.index, inertias["inertia"], linewidth=2, markersize=8)
    plt.xlabel("Anzahl Cluster k", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow-Method", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "elbow_plot.png"), dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_timeseries(
    timeseries: np.ndarray,  # shape: (n_land_cells, 360)
    labels: np.ndarray,  # shape: (n_land_cells,)
    n_clusters: int = 5,
    title: str = None,
    parameter_name: str = "Snow Storage (mm)",
    save_path: str = "./results/clustering",
):
    """
    Plots time series for all clusters with quantile bands

    Arguments:
        timeseries: np.ndarray, shape (n_land_cells, 180) - time series data for all land cells
        labels: np.ndarray, shape (n_land_cells,) - cluster assignments
        n_clusters: int - number of clusters
        parameter_name: str - name of the parameter being plotted
    """
    fig, axes = plt.subplots(
        nrows=n_clusters,
        ncols=1,
        sharey=False,
        sharex=False,
        figsize=(16, 4 * n_clusters),  # ← dynamische Höhe
        constrained_layout=True,  # ← verhindert Überlappung automatisch
    )

    # Falls nur 1 Cluster (sollte nicht passieren), axes in Liste umwandeln
    if n_clusters == 1:
        axes = [axes]

    for cluster in range(n_clusters):
        # Alle Zeitreihen dieses Clusters extrahieren
        cluster_mask = labels == cluster
        cluster_ts = timeseries[cluster_mask]  # shape: (n_cells_in_cluster, 360)

        ax = axes[cluster]

        # Quantile berechnen und plotten
        for q in np.arange(0.1, 0.6, 0.1):
            q_up = np.quantile(cluster_ts, 1 - q, axis=0)
            q_down = np.quantile(cluster_ts, q, axis=0)

            ax.fill_between(
                x=range(180),
                y1=q_down,
                y2=q_up,
                color="#3A6A91",
                alpha=q * 2,
            )

        # Median plotten
        median = np.median(cluster_ts, axis=0)
        ax.plot(range(180), median, color="black", linewidth=2)

        # Labels und Titel
        ax.set_title(
            f"Cluster {cluster}\n({cluster_ts.shape[0]} cells, "
            f"{cluster_ts.shape[0]/len(labels)*100:.1f}%)"
        )
        ax.set_xlabel("Months")

        if cluster == 0:
            ax.set_ylabel(parameter_name)

            # Legende nur im ersten Subplot
            patches_list = []
            patches_list.append(mpatches.Patch(color="black", label="Median"))
            patches_list.append(
                mpatches.Patch(color="#3A6A91", label="Q40 - Q60", alpha=0.8)
            )
            patches_list.append(
                mpatches.Patch(color="#3A6A91", label="Q30 - Q70", alpha=0.6)
            )
            patches_list.append(
                mpatches.Patch(color="#3A6A91", label="Q20 - Q80", alpha=0.4)
            )
            patches_list.append(
                mpatches.Patch(color="#3A6A91", label="Q10 - Q90", alpha=0.2)
            )
            ax.legend(handles=patches_list, loc="best")

        # Grid hinzufügen
        ax.grid(True, alpha=0.3)

    # Subplot-Labels (a), b), c), d))
    for i, ax in enumerate(axes):
        ax.text(
            0.02,
            0.98,
            f"{chr(97+i)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
            color="black",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.savefig(
        os.path.join(save_path, f"timeseries_{n_clusters}_clusters_{title}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_cluster_spatial(
    da: xr.DataArray,
    labels: np.ndarray,  # shape: (n_land_cells,)
    title: str = None,
    n_clusters: int = 5,
    save_path: str = "./results/clustering",
):

    template = da.isel(time=0)

    # Leeres Raster
    cluster_map = xr.full_like(template, fill_value=np.nan)
    land_mask = ~np.isnan(da.isel(time=0))
    cluster_map.values[land_mask.values] = labels

    data = cluster_map

    plt.rcParams.update(
        {
            "axes.edgecolor": "black",
            "axes.linewidth": 0.2,
        }
    )

    # ===== KONFIGURATION =====
    # Farbpalette
    colors_full = [
        # "#e0f3f8",
        # "#abd9e9",  # 1 000–2 000  hellblau
        "#74add1",  # 200–300
        "#313695",
        "#7435BD",
        "#4E2B75",
        "#231334",
    ]  # 400–600
    # Extrahiere eindeutige Klassen aus den xarray-Daten
    unique_classes = np.arange(n_clusters)  # Ignoriere NaN

    # Wähle entsprechende Anzahl Farben
    colors = colors_full[:n_clusters]

    # Erstelle Colormap für diskrete Klassen
    cmap = mcolors.ListedColormap(colors)

    # Bounds für diskrete Klassen
    bounds = np.arange(unique_classes[0] - 0.5, unique_classes[-1] + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n_clusters)

    # ===== PLOT =====
    fig = plt.figure(figsize=(20, 10))
    ax = plt.axes(projection=ccrs.Robinson())

    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=True,
        cbar_kwargs={
            "label": "Spatial Snow Storage Cluster ",
            "ticks": unique_classes,
        },
    )
    ax.coastlines(linewidth=0.5, zorder=11)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
    ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
    if title:
        ax.set_title(f"{title}", fontsize=22)
    else:
        ax.set_title(f"Snow Storage Clusters {scenario}", fontsize=22)

    plt.savefig(
        os.path.join(save_path, f"spatial_map_{n_clusters}_clusters_{title}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_cluster_combined(
    da: xr.DataArray,
    timeseries: np.ndarray,  # shape: (n_land_cells, 180)
    labels: np.ndarray,  # shape: (n_land_cells,)
    n_clusters: int = 5,
    title: str = None,
    parameter_name: str = "Snow Storage (mm)",
    save_path: str = "./results/clustering",
):
    # ── Farbpalette (gemeinsam für Karte & Zeitreihen) ────────────────────────
    colors_full = [
        # "#e0f3f8",
        # "#abd9e9",
        "#74add1",
        "#313695",
        "#7435BD",
        "#4E2B75",
        "#231334",
    ]
    colors = colors_full[:n_clusters]
    cmap = mcolors.ListedColormap(colors)

    unique_classes = np.arange(n_clusters)
    bounds = np.arange(unique_classes[0] - 0.5, unique_classes[-1] + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n_clusters)

    plt.rcParams.update({"axes.edgecolor": "black", "axes.linewidth": 0.2})

    # ── Dynamisches Grid-Layout ───────────────────────────────────────────────
    n_cols = 2
    n_ts_rows = math.ceil(n_clusters / n_cols)  # Anzahl Zeilen für Zeitreihen
    n_rows = 1 + n_ts_rows  # +1 für Kartenzeile

    fig_height = 10 + 4 * n_ts_rows  # Karte ~10, je Zeile ~4
    fig = plt.figure(figsize=(20, fig_height))

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        height_ratios=[2.5] + [1] * n_ts_rows,  # Karte größer als Zeitreihen
        hspace=0.35,
        wspace=0.25,
    )

    # Karte: volle Breite in erster Zeile
    map_ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())

    # Zeitreihen-Axes dynamisch erstellen
    ts_axes = []
    for i in range(n_clusters):
        row = 1 + i // n_cols
        col = i % n_cols
        ts_axes.append(fig.add_subplot(gs[row, col]))

    # Leere überschüssige Ax ausblenden (z.B. 5 Cluster → 1 leere Zelle)
    for j in range(n_clusters, n_ts_rows * n_cols):
        row = 1 + j // n_cols
        col = j % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    # ── Karte (oben) ──────────────────────────────────────────────────────────
    template = da.isel(time=0)
    cluster_map = xr.full_like(template, fill_value=np.nan)
    land_mask = ~np.isnan(da.isel(time=0))
    cluster_map.values[land_mask.values] = labels

    cluster_map.plot(
        ax=map_ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=True,
        cbar_kwargs={"label": "Cluster", "ticks": unique_classes},
    )
    map_ax.coastlines(linewidth=0.5, zorder=11)
    map_ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
    map_ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
    map_ax.set_title(
        title if title else f"Snow Storage Clusters - {scenario} - {property}",
        fontsize=18,
    )

    # ── Zeitreihen (dynamisch darunter) ───────────────────────────────────────
    for cluster in range(n_clusters):
        cluster_mask = labels == cluster
        cluster_ts = timeseries[cluster_mask]
        ax = ts_axes[cluster]
        color = colors[cluster]

        for q in np.arange(0.1, 0.6, 0.1):
            q_up = np.quantile(cluster_ts, 1 - q, axis=0)
            q_down = np.quantile(cluster_ts, q, axis=0)
            ax.fill_between(
                range(timeseries.shape[1]), q_down, q_up, color=color, alpha=q * 2
            )

        median = np.median(cluster_ts, axis=0)
        ax.plot(range(timeseries.shape[1]), median, color="black", linewidth=2)

        ax.set_title(
            f"Cluster {cluster}  ({cluster_ts.shape[0]} cells, "
            f"{cluster_ts.shape[0]/len(labels)*100:.1f}%)",
            fontsize=11,
        )
        ax.set_xlabel("Months")
        ax.set_ylabel(parameter_name)
        ax.grid(True, alpha=0.3)

        # Subplot-Label a), b), …
        ax.text(
            0.02,
            0.98,
            f"{chr(97+cluster)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        # Legende nur im ersten Subplot
        if cluster == 0:
            patches_list = [
                mpatches.Patch(color="black", label="Median"),
                mpatches.Patch(color=color, label="Q40–Q60", alpha=0.8),
                mpatches.Patch(color=color, label="Q30–Q70", alpha=0.6),
                mpatches.Patch(color=color, label="Q20–Q80", alpha=0.4),
                mpatches.Patch(color=color, label="Q10–Q90", alpha=0.2),
            ]
            ax.legend(handles=patches_list, loc="best", fontsize=9)

    # ── Speichern ─────────────────────────────────────────────────────────────
    plt.savefig(
        os.path.join(save_path, f"combined_{n_clusters}_clusters_{title}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":

    base_path = "./results/clustering"

    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_ctrl = xr.open_dataset("./results/Control/snow_control.nc")

    datasets = {
        "47": ds_47,
        "16": ds_16,
        "150": ds_150,
        "Control": ds_ctrl,
        "all": [ds_ctrl, ds_16, ds_47, ds_150],
    }

    for folder in sorted(os.listdir(base_path)):
        folder_path = os.path.join(base_path, folder)

        if not os.path.isdir(folder_path):
            continue

        inertias_file = os.path.join(folder_path, "inertias.csv")

        if not os.path.exists(inertias_file):
            print(f"⚠️  Keine inertias.csv in: {folder}")
            continue

        print(f"{folder}")
        inertias = pd.read_csv(inertias_file, sep=";", index_col=0)
        print("Saving Plot...")
        elbow_plot(inertias, folder_path)

        if folder != "all_dim_22644_180_1":
            # Timeseries Plots
            scenario = folder.split("_")[0]
            property = folder.split("_")[2]
            da_scenario = datasets.get(scenario).snow_storage
            if property == "anomaly":
                da_ctrl = datasets.get("Control").snow_storage
                da = da_scenario - da_ctrl
            else:
                da = da_scenario
            timeseries = prepare_time_series(da)
            timeseries = timeseries.squeeze()

            # Labels einlesen
            for i in range(3, 6):
                labels = np.load(os.path.join(folder_path, f"{i}_cluster_labels.npy"))
                print(f"Dimensionen von labels: {labels.shape}")
                plot_cluster_timeseries(
                    timeseries,
                    labels,
                    n_clusters=i,
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
                plot_cluster_spatial(da, labels, n_clusters=i, save_path=folder_path)
                plot_cluster_combined(
                    da,
                    timeseries,
                    labels,
                    n_clusters=i,
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )

        if folder == "all_dim_22644_180_1":
            folder_path = os.path.join(base_path, folder)
            ts_ctrl = prepare_time_series(datasets["Control"].snow_storage)
            ts_16 = prepare_time_series(datasets["16"].snow_storage)
            ts_47 = prepare_time_series(datasets["47"].snow_storage)
            ts_150 = prepare_time_series(datasets["150"].snow_storage)
            timeseries = np.concatenate([ts_ctrl, ts_16, ts_47, ts_150], axis=0)
            timeseries = timeseries.squeeze()

            for i in range(3, 6):
                labels = np.load(os.path.join(folder_path, f"{i}_cluster_labels.npy"))
                plot_cluster_spatial(
                    datasets.get("Control").snow_storage,
                    labels[0:5661],
                    n_clusters=i,
                    title="Control",
                    save_path=folder_path,
                )
                plot_cluster_spatial(
                    datasets.get("16").snow_storage,
                    labels[5661:11322],
                    n_clusters=i,
                    title="16_Tg",
                    save_path=folder_path,
                )
                plot_cluster_spatial(
                    datasets.get("47").snow_storage,
                    labels[11322:16983],
                    n_clusters=i,
                    title="47_Tg",
                    save_path=folder_path,
                )
                plot_cluster_spatial(
                    datasets.get("150").snow_storage,
                    labels[16983:],
                    n_clusters=i,
                    title="150_Tg",
                    save_path=folder_path,
                )
                plot_cluster_timeseries(
                    timeseries[0:5661],
                    labels[0:5661],
                    n_clusters=i,
                    title="Control",
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
                plot_cluster_timeseries(
                    timeseries[5661:11322],
                    labels[5661:11322],
                    n_clusters=i,
                    title="16_Tg",
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
                plot_cluster_timeseries(
                    timeseries[11322:16983],
                    labels[11322:16983],
                    n_clusters=i,
                    title="47_Tg",
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
                plot_cluster_timeseries(
                    timeseries[16983:22644],
                    labels[16983:22644],
                    n_clusters=i,
                    title="150_Tg",
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
                plot_cluster_timeseries(
                    timeseries,
                    labels,
                    n_clusters=i,
                    title="All_Scenarios",
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
