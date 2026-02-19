import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import xarray as xr


# Funktioniert nur unter der Annahme, dass die labels als Variable im DataSet angelegt sind
# so umschreiben, dass n_clusters aus variablen.attrs ausgelesen wird
def plot_cluster_timeseries(
    ds: xr.Dataset,
    variable: str = "snow_storage",
    cluster_var: str = "snow_cluster",
    n_clusters: int = 5,
    parameter_name: str = "Parameter",
):
    """
    Plot cluster-wise time series including quantile bands and median.

    The function extracts the time series of `variable` from the dataset
    and groups them according to the spatial cluster assignments stored
    in `cluster_var`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to analyse and the cluster labels.
        Required dimensions:
            variable: (time, lat, lon)
            cluster_var: (lat, lon)
    variable : str
        Name of the variable containing time series data.
    cluster_var : str
        Name of the variable containing spatial cluster labels.
    n_clusters : int
        Number of clusters.
    parameter_name : str
        Label used for y-axis description in the plot.

    Notes
    -----
    - Grid cells with NaN cluster labels are ignored.
    - No normalization is applied internally.
    - Quantile bands are shown between:
        Q10–Q90, Q20–Q80, Q30–Q70, Q40–Q60.
    """

    da = ds[variable]
    clusters = ds[cluster_var]

    # Stack spatial dimensions
    da_stacked = da.stack(points=("lat", "lon"))
    clusters_stacked = clusters.stack(points=("lat", "lon"))

    # Remove grid cells without cluster assignment
    valid_mask = clusters_stacked.notnull()
    da_stacked = da_stacked.sel(points=valid_mask)
    clusters_stacked = clusters_stacked.sel(points=valid_mask)

    # Convert to numpy
    timeseries = da_stacked.transpose("points", "time").values
    labels = clusters_stacked.values.astype(int)

    # Create figure
    fig, axes = plt.subplots(
        nrows=n_clusters, ncols=1, sharey=False, sharex=False, figsize=(16, 16)
    )

    if n_clusters == 1:
        axes = [axes]

    for cluster in range(n_clusters):

        cluster_mask = labels == cluster
        cluster_ts = timeseries[cluster_mask]

        ax = axes[cluster]

        if cluster_ts.shape[0] == 0:
            ax.set_title(f"Cluster {cluster} (no grid cells)")
            continue

        # Quantile bands
        for q in np.arange(0.1, 0.6, 0.1):
            q_up = np.quantile(cluster_ts, 1 - q, axis=0)
            q_down = np.quantile(cluster_ts, q, axis=0)

            ax.fill_between(
                x=range(cluster_ts.shape[1]),
                y1=q_down,
                y2=q_up,
                color="#3A6A91",
                alpha=q * 2,
            )

        # Median
        median = np.median(cluster_ts, axis=0)
        ax.plot(range(cluster_ts.shape[1]), median, color="black", linewidth=2)

        ax.set_title(
            f"Cluster {cluster}\n"
            f"({cluster_ts.shape[0]} cells, "
            f"{cluster_ts.shape[0]/len(labels)*100:.1f}%)"
        )
        ax.set_xlabel("Time step")

        if cluster == 0:
            ax.set_ylabel(parameter_name)

            patches_list = [
                mpatches.Patch(color="black", label="Median"),
                mpatches.Patch(color="#3A6A91", label="Q40 - Q60", alpha=0.8),
                mpatches.Patch(color="#3A6A91", label="Q30 - Q70", alpha=0.6),
                mpatches.Patch(color="#3A6A91", label="Q20 - Q80", alpha=0.4),
                mpatches.Patch(color="#3A6A91", label="Q10 - Q90", alpha=0.2),
            ]
            ax.legend(handles=patches_list, loc="best")

        ax.grid(True, alpha=0.3)

    # subplot labels
    for i, ax in enumerate(axes):
        ax.text(
            0.02,
            0.98,
            f"{chr(97+i)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="left",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    plt.tight_layout()
    plt.show()
