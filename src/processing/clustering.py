import xarray as xr
import numpy as np
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMinMax
from datetime import datetime
import os
import pickle
import pandas as pd


def prepare_time_series(da: xr.DataArray) -> np.ndarray:
    """
    Prepares the time series data for tslearn clustering.

    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)

    Returns:
        timeseries_object: np.ndarray,
                           shape (n_land_cells, time_steps, 1)
    """

    # Reorder Dimensions to (lat, lon, time)
    da_t = da.transpose("lat", "lon", "time")

    # Create land mask (assuming NaNs represent ocean)
    land_mask = ~np.isnan(da.isel(time=0))

    # Extract time series for land cells only
    timeseries_2d = da_t.values[land_mask.values]
    # shape: (n_land_cells, time_steps)

    # Add feature dimension for tslearn
    timeseries_3d = timeseries_2d[..., np.newaxis]
    # shape: (n_land_cells, time_steps, 1)

    return timeseries_3d


def time_series_analysis(timeseries_object: np.ndarray, n_clusters: int):
    """
    Does time series analysis on the timeseries objects.
    All the time serieses are clustered based on their
    overall shape using k-means
    Inspired by this article:
    https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook

    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)
        n_clusters: int - the number of clusters to use

    Returns:
        labels: np.ndarray, shape (n_land_cells) - the labels for each time series
        km: TimeSeriesKMeans - the k-means object
    """
    # Normalize the data
    print("Apply MinMax-Scaling method")
    scaler = TimeSeriesScalerMinMax()
    timeseries_scaled = scaler.fit_transform(timeseries_object)

    # Perform time series k-means clustering
    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        n_jobs=-1,  # use all cpu cores
        max_iter=50,
        random_state=42,
        verbose=1,
    )
    labels = km.fit_predict(timeseries_scaled)
    return labels, km


def convert_labels_to_xarray(labels: np.ndarray, da: xr.DataArray) -> xr.DataArray:
    """Adds cluster labels to Dataset
    Arguments:
        labels: np.ndarray, shape (n_land_cells) - the labels for each time series
        da: xr.DataArray - provides shape for the cluster labels
    Returns:
        labels_xarray: xr.DataArray
    """
    labels_xarray = xr.full_like(da.isel(time=0), fill_value=np.nan)
    landmask = ~np.isnan(da.isel(time=0))
    labels_xarray.values[landmask.values] = labels
    return labels_xarray


def elbow_method(timeseries, max_clusters):
    """
    Finds the optimal number of clusters using the elbow method
    https://predictivehacks.com/k-means-elbow-method-code-for-python/
    Arguments:
        data: pandas.DataFrame
        max_clusters: int - the maximum number of clusters to try (inclusive)
    Returns:
        inertias_df: DataFrame with inertias for each k
    """

    # Path to Checkpoint File
    checkpoint_path = os.path.join(
        "results",
        "clustering",
        "snow_storage",
        "elbow_method",
        "150_Tg_euclidean",
        "inertias.csv",
    )

    # Create directory
    cluster_results_dir = os.path.join(
        "results",
        "clustering",
        "snow_storage",
        "elbow_method",
        "150_Tg_euclidean",
    )
    os.makedirs(cluster_results_dir, exist_ok=True)

    # Lade bestehende Ergebnisse falls vorhanden
    if os.path.exists(checkpoint_path):
        print(f"Lade bestehende Ergebnisse aus {checkpoint_path}")
        inertias_df = pd.read_csv(checkpoint_path, sep=";", index_col=0)
        inertias = inertias_df.to_dict()[inertias_df.columns[0]]
        # Konvertiere Keys zu int
        inertias = {int(k): v for k, v in inertias.items()}
        print(f"Bereits berechnet: k = {sorted(inertias.keys())}")
    else:
        inertias = {}

    # Find the optimal number of clusters
    for i in range(2, max_clusters + 1):  # +1 damit max_clusters inklusiv ist!
        if i in inertias:
            print(f"k={i} bereits berechnet, überspringe...")
            continue

        print(f"\n{'='*60}")
        begin = datetime.now()
        print(f"{begin.strftime('%H:%M:%S')} - Trying {i} clusters")
        print(f"{'='*60}")

        labels, km = time_series_analysis(timeseries, i)
        inertias[i] = km.inertia_
        print(f"✓ k={i} fertig! Inertia: {km.inertia_:.2f}")

        # Speichere Labels und Modell
        labels_path = os.path.join(cluster_results_dir, f"{i}_cluster_labels.npy")
        model_path = os.path.join(cluster_results_dir, f"kmeans_model_{i}.pkl")

        np.save(labels_path, labels)  # Labels als NumPy-Array
        print(f"→ Labels gespeichert: {labels_path}")

        with open(model_path, "wb") as f:
            pickle.dump(km, f)  # Modell mit pickle
        print(f"→ Modell gespeichert: {model_path}")

        # Speichere Inertias nach jedem k
        inertias_df = pd.DataFrame.from_dict(
            inertias, orient="index", columns=["inertia"]
        )
        print("schreibe CSV")
        inertias_df.index.name = "k"
        inertias_df.to_csv(checkpoint_path, sep=";")
        print(f"→ Inertias gespeichert: {checkpoint_path}\n")
        print(f"fertig um {datetime.now().strftime('%H:%M:%S')}")

    print(f"\n{'='*60}")
    print("Alle Cluster fertig berechnet!")
    print(f"{'='*60}")

    return inertias_df


def elbow_and_silhouette_method(timeseries, max_clusters):
    """
    Berechnet Elbow-Inertias UND Silhouette Scores für verschiedene k.
    """
    checkpoint_dir = os.path.join(
        "results",
        "clustering",
        "snow_storage",
        "elbow_method",
        "47_Tg_dtw_Subset",  # oder dein Scenario
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_inertias = os.path.join(checkpoint_dir, "inertias.csv")
    checkpoint_silhouettes = os.path.join(checkpoint_dir, "silhouette_scores.csv")

    # Load existing results if available
    if os.path.exists(checkpoint_inertias):
        print(f"Lade bestehende Inertias aus {checkpoint_inertias}")
        inertias_df = pd.read_csv(checkpoint_inertias, sep=";", index_col=0)
        inertias = {
            int(k): v for k, v in inertias_df.to_dict()[inertias_df.columns[0]].items()
        }
    else:
        inertias = {}

    if os.path.exists(checkpoint_silhouettes):
        print(f"Lade bestehende Silhouette Scores aus {checkpoint_silhouettes}")
        silhouettes_df = pd.read_csv(checkpoint_silhouettes, sep=";", index_col=0)
        silhouettes = {
            int(k): v
            for k, v in silhouettes_df.to_dict()[silhouettes_df.columns[0]].items()
        }
    else:
        silhouettes = {}

    # Normalize once
    print("Apply MinMax-Scaling method")
    scaler = TimeSeriesScalerMinMax()
    timeseries_scaled = scaler.fit_transform(timeseries)

    # Try different k values
    for i in range(2, max_clusters + 1):
        if i in inertias and i in silhouettes:
            print(f"k={i} bereits berechnet, überspringe...")
            continue

        print(f"\n{'='*60}")
        begin = datetime.now()
        print(f"{begin.strftime('%H:%M:%S')} - Trying {i} clusters")
        print(f"{'='*60}")

        # Train model
        km = TimeSeriesKMeans(
            n_clusters=i,
            metric="dtw",
            tol=1e-3,  # tolerance for convergence
            n_jobs=-1,
            max_iter=50,
            random_state=42,
            verbose=1,
        )
        labels = km.fit_predict(timeseries_scaled)

        # Save inertia
        inertias[i] = km.inertia_

        # Compute silhouette score
        sil_score = silhouette_score(timeseries_scaled, labels, metric="dtw")
        silhouettes[i] = sil_score

        print(f"✓ k={i} fertig!")
        print(f"  Inertia: {km.inertia_:.2f}")
        print(f"  Silhouette Score: {sil_score:.4f}")

        # Save labels and model
        labels_path = os.path.join(checkpoint_dir, f"{i}_cluster_labels.npy")
        model_path = os.path.join(checkpoint_dir, f"kmeans_model_{i}.pkl")
        np.save(labels_path, labels)
        with open(model_path, "wb") as f:
            pickle.dump(km, f)

        # Save both CSVs after each k
        inertias_df = pd.DataFrame.from_dict(
            inertias, orient="index", columns=["inertia"]
        )
        inertias_df.index.name = "k"
        inertias_df.to_csv(checkpoint_inertias, sep=";")

        silhouettes_df = pd.DataFrame.from_dict(
            silhouettes, orient="index", columns=["silhouette_score"]
        )
        silhouettes_df.index.name = "k"
        silhouettes_df.to_csv(checkpoint_silhouettes, sep=";")

        print(f"→ Ergebnisse gespeichert")
        print(f"fertig um {datetime.now().strftime('%H:%M:%S')}")

    print(f"\n{'='*60}")
    print("Alle Cluster fertig berechnet!")
    print(f"{'='*60}")

    return inertias_df, silhouettes_df


if __name__ == "__main__":
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_control = xr.open_dataset("./results/Control/snow_control.nc")

    da = ds_47.snow_storage
    timeseries = prepare_time_series(da)
    print(f"Shape of Timeseries: {timeseries.shape}")
    # Subset
    subset_size = int(0.3 * timeseries.shape[0])
    indices = np.random.choice(timeseries.shape[0], subset_size, replace=False)
    timeseries_subset = timeseries[indices]
    print(f"Shape of Subset: {timeseries_subset.shape}")

    elbow_and_silhouette_method(timeseries_subset, max_clusters=10)
