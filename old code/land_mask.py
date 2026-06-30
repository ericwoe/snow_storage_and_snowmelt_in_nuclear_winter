import xarray as xr
import numpy as np
from shapely.geometry import box


def create_mask(da, shape):
    """
    Calculates the land covered fraction for each grid cell.

    Parameter
    ----------
    da : xarray.DataArray
        Containing lat and lon coordinates defining the grid for which
        the land fraction should be calculated.
    shape : geopandas.GeoDataFrame
        Shapefile with land polygons (lon -180 bis 180).

    Returns
    -------
    fractions : xarray.DataArray
        2D DataArray (lat, lon) with land fraction between 0 and 1.
    """
    # Combine all land polygons into one geometry (union)
    shape_union = shape.union_all()

    # Initialize result array with zeros, same shape as input dataarray (lat, lon)
    fractions = np.zeros((len(da.lat), len(da.lon)))

    # Calculate half the grid cell size in lat and lon direction to define cell boundaries around
    # the center point
    dlat = float(abs(da.lat[1] - da.lat[0])) / 2
    dlon = float(abs(da.lon[1] - da.lon[0])) / 2

    for i, lat in enumerate(da.lat.values):
        for j, lon in enumerate(da.lon.values):

            # Correct the longitude from 0–360 to -180–180 so it fits the
            # coordinate system of the shapefile
            # Example: 270° -> -90°,  180° -> 180°
            lon_centered = lon if lon <= 180 else lon - 360

            # Calculate cell boundaries in lat direction and clip to ±90°
            # to avoid invalid geometries at the poles
            lat_lower = max(lat - dlat, -90)
            lat_upper = min(lat + dlat, 90)

            # Calculate cell boundaries in lon direction
            lon_left = lon_centered - dlon
            lon_right = lon_centered + dlon

            # Special Case One: Cell crosses the International Date Line (180°) to the right
            # Cell is split into two halves: right of 180° is
            # mirrored to the left side of the map (-180° to lon_right-360°)
            if lon_right > 180:
                cell_left = box(lon_left, lat_lower, 180, lat_upper)
                cell_right = box(-180, lat_lower, lon_right - 360, lat_upper)
                intersection = (
                    shape_union.intersection(cell_left).area
                    + shape_union.intersection(cell_right).area
                )
                cell_area = cell_left.area + cell_right.area

            # Special Case Two: Cell crosses the International Date Line (-180°) to the left
            # analog: left half of the cell is mirrored to the
            # right side of the map (180° to lon_left+360°)
            elif lon_left < -180:
                cell_left = box(lon_left + 360, lat_lower, 180, lat_upper)
                cell_right = box(-180, lat_lower, lon_right, lat_upper)
                intersection = (
                    shape_union.intersection(cell_left).area
                    + shape_union.intersection(cell_right).area
                )
                cell_area = cell_left.area + cell_right.area

            # Usual Case: Cell lies entirely within ±180°
            else:
                cell = box(lon_left, lat_lower, lon_right, lat_upper)
                intersection = shape_union.intersection(cell).area
                cell_area = cell.area

            fractions[i, j] = intersection / cell_area

    # Return as DataArray with same coordinates and dimensions as input dataarray
    return xr.DataArray(
        fractions,
        coords={"lat": da.lat, "lon": da.lon},
        dims=["lat", "lon"],
        attrs={"long_name": "Land fraction", "units": "None"},
    )
