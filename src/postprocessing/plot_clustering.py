import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
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
from src.postprocessing.snow_analysis import compute_grid_cell_area
from matplotlib.ticker import MultipleLocator
import matplotlib.lines as mlines

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> np.ndarray:
    """
    values:  shape (n_cells, time_steps) - Zeitreihenwerte pro Zelle
    weights: shape (n_cells,)            - Flächengewicht pro Zelle
    quantile: float                      - gewünschtes Quantil, z.B. 0.5 für Median
    """

    # -------------------------------------------------------------------------
    # SCHRITT 1: Sortieren
    # -------------------------------------------------------------------------
    # np.argsort gibt nicht die sortierten Werte zurück, sondern die INDIZES
    # die die Werte sortieren würden. axis=0 bedeutet: spaltenweise, also
    # pro Zeitschritt unabhängig sortieren.
    #
    # Beispiel für einen Zeitschritt:
    #   Werte:   [5.2, 2.1, 8.4, 1.3]
    #   Indizes: [3,   1,   0,   2  ]  ← so muss man umsortieren für aufsteig. Reihenfolge
    sorted_indices = np.argsort(values, axis=0)  # shape: (n_cells, time_steps)

    # Mit take_along_axis werden die Werte anhand der Sortier-Indizes umgeordnet.
    # Ergebnis: pro Zeitschritt sind die Zellwerte aufsteigend sortiert.
    #
    #   Vorher:  [5.2, 2.1, 8.4, 1.3]
    #   Nachher: [1.3, 2.1, 5.2, 8.4]
    sorted_values = np.take_along_axis(
        values, sorted_indices, axis=0
    )  # shape: (n_cells, time_steps)

    # Die Gewichte (Flächen) werden in dieselbe Reihenfolge gebracht wie die Werte.
    # weights hat shape (n_cells,), sorted_indices hat shape (n_cells, time_steps).
    # NumPy broadcasted weights automatisch auf alle Zeitschritte.
    #
    # Wichtig: Gewichte müssen zur selben Zelle gehören wie der Wert!
    #   Werte sortiert:   [1.3,  2.1,  5.2,  8.4 ]  ← Zellen 3, 1, 0, 2
    #   Gewichte sortiert:[150,  200,  100,   50  ]  ← Flächen derselben Zellen
    sorted_weights = weights[sorted_indices]  # shape: (n_cells, time_steps)

    # -------------------------------------------------------------------------
    # SCHRITT 2: Kumulative Gewichte berechnen und normalisieren
    # -------------------------------------------------------------------------
    # np.cumsum summiert die Gewichte von oben nach unten (axis=0 = spaltenweise).
    # Dadurch entsteht eine aufsteigende Kurve der "aufgesammelten Fläche".
    #
    #   Gewichte:          [ 150,  200,  100,   50]
    #   Kumulativ:         [ 150,  350,  450,  500]
    cumulative_weights = np.cumsum(
        sorted_weights, axis=0
    )  # shape: (n_cells, time_steps)

    # Durch Division durch den letzten Wert (= Gesamtfläche) wird auf [0, 1] normalisiert.
    # cumulative_weights[-1, :] ist die Gesamtsumme aller Gewichte pro Zeitschritt.
    #
    #   Kumulativ normalisiert: [0.30, 0.70, 0.90, 1.00]
    #
    # Interpretation: Nach der ersten Zelle sind 30% der Gesamtfläche "abgedeckt",
    # nach der zweiten 70%, usw.
    # Jede Fläche als Bruch an Gesamtfäche (zum Ablesen)
    cumulative_weights /= cumulative_weights[-1, :]  # shape: (n_cells, time_steps)

    # -------------------------------------------------------------------------
    # SCHRITT 3: Quantil per Interpolation ablesen
    # -------------------------------------------------------------------------
    # Für jeden Zeitschritt separat: np.interp sucht auf der normierten Flächen-
    # Achse (x) die Position des gewünschten Quantils und liest den zugehörigen
    # Wert (y) ab. Zwischen zwei Stützpunkten wird linear interpoliert.
    #
    # Beispiel für quantile=0.5 (Median):
    #   x (kum. Fläche): [0.30, 0.70, 0.90, 1.00]
    #   y (Werte):       [1.3,  2.1,  5.2,  8.4 ]
    #
    #   np.interp(0.5, x, y) → 0.5 liegt zwischen 0.30 und 0.70
    #   → linear interpoliert: 1.3 + (2.1 - 1.3) * (0.5 - 0.30) / (0.70 - 0.30)
    #   → Ergebnis: ~1.7
    #
    # Ersten Index finden, bei dem die kumulative Fläche >= quantile ist
    # np.argmax gibt den ersten True-Wert zurück (axis=0 = pro Zeitschritt)
    indices = np.argmax(cumulative_weights >= quantile, axis=0)  # shape: (time_steps,)

    # Den zugehörigen Wert aus sorted_values auslesen
    result = sorted_values[indices, np.arange(values.shape[1])]

    return result  # shape: (time_steps,)


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
    cell_areas: np.ndarray,  # shape: (n_land_cells,)
    fractions: np.ndarray,  # shape: (n_land_cells,)
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
        cell_areas: np.ndarray, shape (n_land_cells,) - cell areas for weighted statistics
        n_clusters: int - number of clusters
        parameter_name: str - name of the parameter being plotted
    """
    fig, axes = plt.subplots(
        nrows=n_clusters,
        ncols=1,
        sharey=False,
        sharex=False,
        figsize=(16, 4 * n_clusters),
        constrained_layout=True,
    )

    if n_clusters == 1:
        axes = [axes]

    for cluster in range(n_clusters):
        cluster_mask = labels == cluster
        cluster_ts = timeseries[cluster_mask]  # shape: (n_cells_in_cluster, time)
        # Effektive Fläche = Zellfläche * Landanteil
        effective_areas = cell_areas * fractions  # ← neu
        cluster_weights = effective_areas[cluster_mask]  # shape: (n_cells_in_cluster,)
        ax = axes[cluster]

        for q in np.arange(0.1, 0.6, 0.1):
            q_up = weighted_quantile(cluster_ts, cluster_weights, 1 - q)
            q_down = weighted_quantile(cluster_ts, cluster_weights, q)
            ax.fill_between(
                x=range(cluster_ts.shape[1]),
                y1=q_down,
                y2=q_up,
                color="#3A6A91",
                alpha=q * 2,
            )

        # Gewichteter Median
        median = weighted_quantile(cluster_ts, cluster_weights, 0.5)
        ax.plot(range(cluster_ts.shape[1]), median, color="black", linewidth=2)
        cluster_area_fraction = cluster_weights.sum() / cell_areas.sum() * 100
        ax.set_title(
            f"Cluster {cluster}\n({cluster_area_fraction:.1f}% of total area) - {cluster_ts.shape[0]} cells"
        )
        ax.set_xlabel("Months")

        if cluster == 0:
            ax.set_ylabel("Snow Storage [mm]")
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

        ax.grid(True, alpha=0.3)

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


def plot_cluster_timeseries_ctrl(
    timeseries: np.ndarray,
    timeseries_ctrl: np.ndarray,  # neu: shape (n_land_cells, 180)
    labels: np.ndarray,
    cell_areas: np.ndarray,
    fractions: np.ndarray,
    n_clusters: int = 5,
    title: str = None,
    parameter_name: str = "Snow Storage (mm)",
    save_path: str = "./results/clustering",
):
    import matplotlib.lines as mlines

    fig, axes = plt.subplots(
        nrows=n_clusters,
        ncols=1,
        sharey=False,
        sharex=False,
        figsize=(16, 4 * n_clusters),
        constrained_layout=True,
    )

    if n_clusters == 1:
        axes = [axes]

    for cluster in range(n_clusters):
        cluster_mask = labels == cluster
        cluster_ts = timeseries[cluster_mask]
        cluster_ts_ctrl = timeseries_ctrl[cluster_mask]  # neu

        effective_areas = cell_areas * fractions
        cluster_weights = effective_areas[cluster_mask]

        ax = axes[cluster]

        for q in np.arange(0.1, 0.6, 0.1):
            q_up = weighted_quantile(cluster_ts, cluster_weights, 1 - q)
            q_down = weighted_quantile(cluster_ts, cluster_weights, q)
            ax.fill_between(
                x=range(cluster_ts.shape[1]),
                y1=q_down,
                y2=q_up,
                color="#3A6A91",
                alpha=q * 2,
            )

        # Gewichteter Median Szenario
        median = weighted_quantile(cluster_ts, cluster_weights, 0.5)
        ax.plot(
            range(cluster_ts.shape[1]),
            median,
            color="black",
            linewidth=2,
            label="Median 47 Tg",
        )

        # Gewichteter Median Control – neu
        median_ctrl = weighted_quantile(cluster_ts_ctrl, cluster_weights, 0.5)
        ax.plot(
            range(cluster_ts_ctrl.shape[1]),
            median_ctrl,
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="Median Control",
        )

        cluster_area_fraction = cluster_weights.sum() / cell_areas.sum() * 100
        ax.set_title(
            f"Cluster {cluster}\n({cluster_area_fraction:.1f}% of total area) - {cluster_ts.shape[0]} cells"
        )
        ax.set_xlabel("Months")

        if cluster == 0:
            ax.set_ylabel("Snow Storage (mm)")
            patches_list = []
            patches_list.append(
                mlines.Line2D(
                    [], [], color="#E34444", linestyle="--", label="Median Control"
                )
            )
            patches_list.append(mpatches.Patch(color="black", label="Median 47 Tg"))
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

        ax.grid(True, alpha=0.3)

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
        os.path.join(save_path, f"timeseries_{n_clusters}_clusters_{title}_ctrl.png"),
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
    timeseries_scenario: np.ndarray,
    timeseries_ctrl: np.ndarray,
    labels: np.ndarray,
    cell_areas: np.ndarray,
    fractions: np.ndarray,
    n_clusters: int = 5,
    title: str = None,
    parameter_name: str = "Snow Storage (mm)",
    save_path: str = "./results/clustering",
):
    # ── Farbpalette nach Schneemenge sortieren ────────────────────────────────
    colors_full = ["#f1eef6", "#abd9e9", "#74add1", "#4575b4", "#542788"]
    color = "#4575b4"
    # "#f1eef6", "#abd9e9", "#74add1", "#4575b4", "#542788"

    effective_areas = cell_areas * fractions

    # Mittlere Schneemenge pro Cluster berechnen und sortieren
    cluster_means = {}
    for cluster in range(n_clusters):
        cluster_mask = labels == cluster
        cluster_ts = timeseries_scenario[cluster_mask]
        cluster_means[cluster] = cluster_ts.mean()

    # Cluster nach Schneemenge sortieren (aufsteigend = wenig → viel)
    sorted_clusters = sorted(cluster_means, key=lambda x: cluster_means[x])
    label_mapping = {old: new for new, old in enumerate(sorted_clusters)}
    labels_remapped = np.array([label_mapping[l] for l in labels])

    colors = colors_full[:n_clusters]
    cmap = mcolors.ListedColormap(colors)
    unique_classes = np.arange(n_clusters)
    bounds = np.arange(unique_classes[0] - 0.5, unique_classes[-1] + 1.5, 1)
    norm = mcolors.BoundaryNorm(bounds, ncolors=n_clusters)

    # ── Dynamisches Grid-Layout ───────────────────────────────────────────────
    n_cols = 2
    n_ts_rows = math.ceil(n_clusters / n_cols)
    n_rows = 1 + n_ts_rows

    fig_height = 10 + 4 * n_ts_rows
    fig = plt.figure(figsize=(20, fig_height))

    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=n_cols,
        height_ratios=[2.5] + [1] * n_ts_rows,
        hspace=0.35,
        wspace=0.25,
    )

    map_ax = fig.add_subplot(gs[0, :], projection=ccrs.Robinson())

    ts_axes = []
    for i in range(n_clusters):
        row = 1 + i // n_cols
        col = i % n_cols
        ts_axes.append(fig.add_subplot(gs[row, col]))

    for j in range(n_clusters, n_ts_rows * n_cols):
        row = 1 + j // n_cols
        col = j % n_cols
        fig.add_subplot(gs[row, col]).set_visible(False)

    # ── Karte mit remapped Labels ─────────────────────────────────────────────
    template = da.isel(time=0)
    cluster_map = xr.full_like(template, fill_value=np.nan)
    land_mask = ~np.isnan(da.isel(time=0))
    cluster_map.values[land_mask.values] = labels_remapped  # ← remapped

    cluster_map.plot(
        ax=map_ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=True,
        cbar_kwargs={"label": "Cluster", "ticks": unique_classes},
    )
    map_ax.set_title("")
    map_ax.coastlines(linewidth=0.5, zorder=11)
    map_ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
    map_ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
    """ map_ax.set_title(
        title if title else f"Spatial Distribution of Snow Storage Clusters - 47 Tg",
        fontsize=18,
    )"""

    # ── Zeitreihen nach remapped Labels ──────────────────────────────────────
    # Iteriere über neue Cluster-Reihenfolge (sortiert nach Schneemenge)
    for new_cluster_idx in range(n_clusters):
        # Originalcluster der diesem neuen Index entspricht
        old_cluster = sorted_clusters[new_cluster_idx]
        cluster_mask = labels == old_cluster
        cluster_ts = timeseries_scenario[cluster_mask]
        cluster_ts_ctrl = timeseries_ctrl[cluster_mask]
        cluster_weights = effective_areas[cluster_mask]

        ax = ts_axes[new_cluster_idx]

        q_up_overall = 0
        for q in np.arange(0.1, 0.6, 0.1):
            q_up = weighted_quantile(cluster_ts, cluster_weights, 1 - q)
            q_down = weighted_quantile(cluster_ts, cluster_weights, q)
            q_up_overall = max(q_up_overall, q_up.max())
            ax.fill_between(
                range(cluster_ts.shape[1]),
                q_down,
                q_up,
                color=color,
                alpha=q * 2,
            )
        y_offset = -0.03 * (q_up_overall + 100)  # 3% der Gesamtspanne
        ax.set_ylim(y_offset, q_up_overall + 100)
        ax.set_ylabel("Snow Storage [mm]")

        median = weighted_quantile(cluster_ts, cluster_weights, 0.5)
        ax.plot(
            range(cluster_ts.shape[1]),
            median,
            color="black",
            linewidth=2,
            label="Median Scenario",
        )

        median_ctrl = weighted_quantile(cluster_ts_ctrl, cluster_weights, 0.5)
        ax.plot(
            range(cluster_ts_ctrl.shape[1]),
            median_ctrl,
            color="#E34444",
            linewidth=1.2,
            linestyle="--",
            label="Median Control",
        )

        cluster_area_fraction = cluster_weights.sum() / effective_areas.sum() * 100
        ax.set_title(
            f"Cluster: {new_cluster_idx}, {cluster_area_fraction:.1f}% of Land Area",  # - {cluster_ts.shape[0]} cells
            size=13,
        )

        jan_ticks = np.arange(0, cluster_ts.shape[1], 12)
        ax.set_xticks(jan_ticks)
        ax.set_xticklabels(
            [f"Year {i}" for i in range(len(jan_ticks))],
            rotation=45,
            size=7,
            ha="right",
        )
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel("Time")

        map_ax.text(
            -0.05,
            0.95,
            "a)",
            transform=map_ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        ax.text(
            -0.1,
            1.09,
            f"{chr(98 + new_cluster_idx)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        if new_cluster_idx == 0:
            patches_list = [
                mlines.Line2D([], [], color="black", label="Median Scenario"),
                mpatches.Patch(color=color, label="Q40–Q60", alpha=0.8),
                mpatches.Patch(color=color, label="Q30–Q70", alpha=0.6),
                mpatches.Patch(color=color, label="Q20–Q80", alpha=0.4),
                mpatches.Patch(color=color, label="Q10–Q90", alpha=0.2),
                mlines.Line2D(
                    [], [], color="#E34444", linestyle="--", label="Median Control"
                ),
            ]
            ax.legend(handles=patches_list, loc="best", fontsize=9)

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

    fraction_mask = xr.open_dataarray("./data/interim/land_mask_neu.nc")
    # Calculate cell area for weighting
    cell_area = compute_grid_cell_area(ds_47.snow_storage)
    # Create land mask for extracting cell areas of only land cells
    land_mask = ~np.isnan(ds_47.snow_storage.isel(time=0))
    cell_area_1d = cell_area.values[land_mask.values]  # shape: (n_land_cells,)
    fractions_1d = fraction_mask.values[land_mask.values]  # shape: (n_land_cells,)

    timeseries_ctrl = prepare_time_series(ds_ctrl.snow_storage)
    timeseries_ctrl = timeseries_ctrl.squeeze()

    timeseries_scenario = prepare_time_series(ds_47.snow_storage)
    timeseries_scenario = timeseries_scenario.squeeze()

    labels = np.load("./results/clustering/47_Tg_dtw/5_cluster_labels.npy")

    plot_cluster_combined(
        da=ds_47.snow_storage,
        timeseries_scenario=timeseries_scenario,
        timeseries_ctrl=timeseries_ctrl,
        labels=labels,
        cell_areas=cell_area_1d,
        fractions=fractions_1d,
        n_clusters=5,
        title=None,
        parameter_name="Snow Storage (mm)",
        save_path="./results/clustering",
    )

    """for folder in sorted(os.listdir(base_path)):
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
                    cell_area_1d,
                    fractions_1d,
                    n_clusters=i,
                    parameter_name="Snow Storage (mm)",
                    save_path=folder_path,
                )
                plot_cluster_timeseries(
                    timeseries,
                    timeseries_ctrl,
                    labels,
                    cell_area_1d,
                    fractions_1d,
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
                )"""
