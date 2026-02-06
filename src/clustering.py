import xarray as xr
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler


def prepare_time_series(da: xr.DataArray) -> np.ndarray:
    """
    Prepares the time series data for clustering
    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)
    Returns:
        X: np.ndarray, shape (n_land_cells, time_steps)
    """
    # Transpose to (lat, lon, time)
    da_t = da.transpose("lat", "lon", "time")

    # Create a land mask (assuming NaNs represent ocean)
    land_mask = ~np.isnan(da.isel(time=0))

    # Extract time series for land cells only
    timeseries_object = da_t.values[land_mask.values]  # shape: (n_land_cells, time)
    return timeseries_object


def time_series_analysis(timeseries_object: np.ndarray, n_clusters):
    """
    Does time series analysis on the dataarray
    All the time serieses are clustered based on their
    overall shape using k-means
    Inspired by this article:
    https://www.kaggle.com/code/izzettunc/introduction-to-time-series-clustering/notebook
    Arguments:
        da: xarray.DataArray, shape (time, lat, lon)
        n_clusters: int - the number of clusters to use
    Returns:
        labels: list - the labels for each time series
        km: TimeSeriesKMeans - the k-means object
    """
    # Normalize the data
    scaler = MinMaxScaler()
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
