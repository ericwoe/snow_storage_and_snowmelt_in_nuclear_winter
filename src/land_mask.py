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
    # Lokale Kopie für Berechnung, Originaldataset bleibt unverändert
    ds_temp = ds.copy()

    # Temporär auf [-180, 180] bringen für Spatial Join
    ds_temp = ds_temp.assign_coords(lon=((ds_temp.lon + 180) % 360) - 180).sortby("lon")

    lat = ds_temp.lat.values
    lon = ds_temp.lon.values
    res_lat = lat[1] - lat[0]
    res_lon = lon[1] - lon[0]

    # Rasterzellenpolygone erstellen
    polygons = [
        box(x - res_lon / 2, y - res_lat / 2, x + res_lon / 2, y + res_lat / 2)
        for y in lat
        for x in lon
    ]

    cells_gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    # GADM-Landpolygone laden
    gadm = gpd.read_file(gadm_path)

    # Spatial join: nur Zellen, die irgendeinem Landpolygon schneiden
    cells_land = gpd.sjoin(cells_gdf, gadm, how="inner", predicate="intersects")

    # Boolean-Maske erstellen
    land_mask_flat = np.zeros(len(polygons), dtype=bool)
    land_mask_flat[cells_land.index.values] = True
    land_mask = land_mask_flat.reshape(len(lat), len(lon))

    land_mask_xr = xr.DataArray(
        land_mask,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon"),
        name="land_mask",
    )

    # Longitude wieder auf 0-360 transformieren für einfache Verwendung
    land_mask_xr = land_mask_xr.assign_coords(
        lon=((land_mask_xr.lon + 360) % 360)
    ).sortby("lon")

    return land_mask_xr


def save_land_mask(land_mask: xr.DataArray, path: str):
    """
    Save land mask DataArray to NetCDF.

    Parameters
    ----------
    land_mask : xr.DataArray
        Boolean land mask to save
    path : str
        Path to NetCDF file
    """
    land_mask.to_netcdf(path, mode="w", format="NETCDF4")


if __name__ == "__main__":
    # Land mask
    landmask = land_mask.create_land_mask(ds, gadm_path)
    land_mask.save_land_mask(landmask, land_mask_path)
