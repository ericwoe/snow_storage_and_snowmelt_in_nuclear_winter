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
