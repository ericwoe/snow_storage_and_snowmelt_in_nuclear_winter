import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import numpy as np


def create_land_mask(ds: xr.Dataset, gadm_path: str) -> xr.DataArray:
    """
    Create a boolean land mask from a GADM shapefile for the lat/lon grid of a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'lat' and 'lon' coordinates
    gadm_path : str
        File path to GADM shapefile or GeoPackage

    Returns
    -------
    land_mask_xr : xr.DataArray
        Boolean land mask (True = land) with same lat/lon as dataset.
        Longitude is returned in 0-360 range for einfache Verwendung.
    """
    # local copy of dataset
    ds_temp = ds.copy()

    # change longitude from [0, 360] to [-180, 180] for spatial join
    ds_temp = ds_temp.assign_coords(lon=((ds_temp.lon + 180) % 360) - 180).sortby("lon")

    lat = ds_temp.lat.values
    lon = ds_temp.lon.values
    res_lat = lat[1] - lat[0]
    res_lon = lon[1] - lon[0]

    # create grid cell polygons
    polygons = [
        box(x - res_lon / 2, y - res_lat / 2, x + res_lon / 2, y + res_lat / 2)
        for y in lat
        for x in lon
    ]

    cells_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # import gadm land area polygons
    gadm = gpd.read_file(gadm_path)

    # Spatial join: only cell polygons that intersect with land area polygons
    cells_land = gpd.sjoin(cells_gdf, gadm, how="inner", predicate="intersects")

    # create boolean land mask
    land_mask_flat = np.zeros(len(polygons), dtype=bool)
    land_mask_flat[cells_land.index.values] = True
    land_mask = land_mask_flat.reshape(len(lat), len(lon))

    land_mask_xr = xr.DataArray(
        land_mask,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="land_mask",
    )

    # bring longitude back to [0, 360] range
    land_mask_xr = land_mask_xr.assign_coords(
        lon=((land_mask_xr.lon + 360) % 360)
    ).sortby("lon")

    land_mask_xr.attrs = {
        "long_name": "Land area mask",
        "standard_name": "land_binary_mask",
        "description": "Boolean mask indicating land grid cells (True=land, False=ocean/no data)",
        "method": "Spatial intersection with GADM polygons",
        "crs": "EPSG:4326",
        "valid_range": "0, 1",
    }

    return land_mask_xr
