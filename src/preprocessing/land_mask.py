import xarray as xr
import geopandas as gpd
import numpy as np


def create_mask(ds: xr.Dataset, shape: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Create a fractional mask from a shapefile for the lat/lon grid of a dataset.

    For each grid cell, the fraction of the cell area that intersects with the
    provided shape is computed. This allows for partial coverage of coastal or
    boundary cells, rather than a binary land/ocean assignment.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'lat' and 'lon' coordinates in [0, 360] range.
    shape : gpd.GeoDataFrame
        GeoDataFrame containing the shape polygons to mask against (e.g. land area,
        country boundaries). Must be in EPSG:4326 (WGS84).

    Returns
    -------
    xr.DataArray
        Fractional mask with values in [0.0, 1.0], where 0.0 means no overlap
        with the shape and 1.0 means full coverage. Has the same lat/lon grid
        as the input dataset. Longitude is returned in [0, 360] range.

    Notes
    -----
    - Longitude is temporarily converted to [-180, 180] for the spatial
      intersection, then converted back to [0, 360].
    - Area fractions are computed in geographic coordinates (degrees), not in
      an equal-area projection. This introduces a small distortion at high
      latitudes, which is acceptable for most masking applications.
    - All shape polygons are merged into a single union before intersection,
      which avoids double-counting of overlapping polygons.

    Examples
    --------
    >>> land_mask = create_mask(ds, gadm_gdf)
    >>> snow_volume = ds.snow_storage * cell_area * land_mask
    """
    from shapely.geometry import box

    ds_temp = ds.copy()
    ds_temp = ds_temp.assign_coords(lon=((ds_temp.lon + 180) % 360) - 180).sortby("lon")
    lat = ds_temp.lat.values
    lon = ds_temp.lon.values
    res_lat = lat[1] - lat[0]
    res_lon = lon[1] - lon[0]

    polygons = [
        box(x - res_lon / 2, y - res_lat / 2, x + res_lon / 2, y + res_lat / 2)
        for y in lat
        for x in lon
    ]
    cells_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    shape_union = shape.union_all()

    fractional_coverage = np.zeros(len(polygons), dtype=float)
    for i, cell in enumerate(cells_gdf.geometry):
        if cell.intersects(shape_union):
            intersection = cell.intersection(shape_union)
            fractional_coverage[i] = intersection.area / cell.area

    land_mask = fractional_coverage.reshape(len(lat), len(lon))
    land_mask_xr = xr.DataArray(
        land_mask,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="land_mask",
    )
    land_mask_xr = land_mask_xr.assign_coords(
        lon=((land_mask_xr.lon + 360) % 360)
    ).sortby("lon")

    return land_mask_xr
