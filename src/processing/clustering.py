import xarray as xr
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax


def prepare_time_series(da: xr.DataArray) -> np.ndarray:
    """
    Prepares the time series data for tslearn clustering.

    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)

    Returns:
        timeseries_object: np.ndarray,
                           shape (n_land_cells, time_steps, 1)
    """

    # Transpose to (lat, lon, time)
    da_t = da.transpose("lat", "lon", "time")

    # Create land mask (assuming NaNs represent ocean)
    land_mask = ~np.isnan(da.isel(time=0))

    # Extract time series for land cells
    timeseries_2d = da_t.values[land_mask.values]
    # shape: (n_land_cells, time_steps)

    # Add feature dimension for tslearn
    timeseries_3d = timeseries_2d[..., np.newaxis]
    # shape: (n_land_cells, time_steps, 1)

    return timeseries_3d


def time_series_analysis(timeseries_object: np.ndarray, n_clusters):
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
    print("Apply new scaling method")
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
        cluster_array: xr.DataArray
    """
    cluster_array = xr.full_like(da, fill_value=np.nan)
    landmask = ~np.isnan(da.isel(time=0))
    cluster_array.values[landmask.values] = labels
    return cluster_array


'''def cluster_snow_timeseries(snow: xr.DataArray, n_clusters: int = 4):
    """
    Cluster complete snow storage time series for each spatial grid cell.

    This function performs a clustering analysis on the full temporal
    evolution of `snow_storage` at each (lat, lon) grid cell. Only grid
    cells with completely valid (NaN-free) time series are included in
    the clustering. Grid cells containing NaN values over time are
    excluded and returned as NaN in the output.

    Each grid cell is treated as one sample and the time dimension
    represents the feature space. Clustering is performed using
    k-means.

    Parameters
    ----------
    snow : xr.DataArray
        Snow storage data with dimensions (time, lat, lon).
        The time series at each grid cell is used as feature vector.
    n_clusters : int, optional
        Number of clusters for the k-means algorithm (default is 4).

    Returns
    -------
    xr.DataArray
        Cluster labels with dimensions (lat, lon). Grid cells that were
        excluded due to missing values are set to NaN. Valid grid cells
        contain integer cluster IDs in the range [0, n_clusters-1].

    Notes
    -----
    - No standardization of time series is applied. Therefore, clustering
      reflects similarities in both magnitude and temporal dynamics.
    - If clustering should be based on temporal shape only, the input
      data should be standardized prior to clustering.
    - The function performs no file I/O and does not modify the input
      dataset.
    """

    # Stack spatial dims
    snow_2d = snow.stack(points=("lat", "lon"))

    # Mask: only points without NaNs over dimension time 
    valid_mask = snow_2d.notnull().all(dim="time")

    # only keep valid points
    snow_valid = snow_2d.sel(points=valid_mask)

    # Feature-Matrix 
    X = snow_valid.transpose("points", "time").values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    km = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        n_jobs=-1,  # use all cpu cores
        max_iter=50,
        random_state=42,
        verbose=1,
    labels = km.fit_predict(X_scaled)

    # DataArray mit nur validen Punkten
    clusters_valid = xr.DataArray(
        labels,
        coords={"points": snow_valid.points},
        dims=("points",),
        name="snow_cluster",
    )

    # Leeres Array mit NaN vorbereiten
    clusters_full = xr.full_like(snow_2d.isel(time=0), fill_value=np.nan)
    clusters_full.loc[dict(points=clusters_valid.points)] = clusters_valid
    clusters_full = clusters_full.unstack("points")
    return clusters_full'''
