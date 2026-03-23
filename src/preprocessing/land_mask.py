import xarray as xr
import geopandas as gpd
import numpy as np
from shapely.geometry import box


def create_mask(da, shape):
    """
    Berechnet den Landanteil jeder Rasterzelle.
    Parameter
    ----------
    da : xarray.DataArray
        Muss lat- und lon-Koordinaten besitzen (lon 0–360).
    shape : geopandas.GeoDataFrame
        Shapefile mit Landpolygonen (lon -180 bis 180).
    Returns
    -------
    fractions : xarray.DataArray
        2D DataArray (lat, lon) mit Landanteilen zwischen 0 und 1.
    """
    # Alle Landpolygone zu einem einzigen Polygon zusammenführen
    # → vermeidet mehrfache Verschneidungen pro Zelle
    shape_union = shape.union_all()

    # Ergebnisarray initialisieren (alle Werte = 0)
    fractions = np.zeros((len(da.lat), len(da.lon)))

    # Halbe Zellgröße in jede Richtung berechnen
    # → wird benötigt um die Zellgrenzen um den Mittelpunkt zu berechnen
    dlat = float(abs(da.lat[1] - da.lat[0])) / 2
    dlon = float(abs(da.lon[1] - da.lon[0])) / 2

    for i, lat in enumerate(da.lat.values):
        for j, lon in enumerate(da.lon.values):

            # Lon von 0–360 auf -180–180 umrechnen damit es zum
            # Koordinatensystem des Shapefiles passt
            # Beispiel: 270° → -90°,  180° bleibt 180°
            lon_centered = lon if lon <= 180 else lon - 360

            # Zellgrenzen in Lat-Richtung berechnen und auf ±90° clippen
            # → verhindert dass Polzellen über den Pol hinausragen
            lat_lower = max(lat - dlat, -90)
            lat_upper = min(lat + dlat, 90)

            # Zellgrenzen in Lon-Richtung berechnen
            lon_left = lon_centered - dlon
            lon_right = lon_centered + dlon

            # Sonderfall: Zelle überschreitet den Antimeridian (180°) nach rechts
            # → Zelle wird in zwei Hälften aufgeteilt: rechts von 180° wird
            #    auf die linke Seite der Karte gespiegelt (-180° bis lon_right-360°)
            if lon_right > 180:
                cell_left = box(lon_left, lat_lower, 180, lat_upper)
                cell_right = box(-180, lat_lower, lon_right - 360, lat_upper)
                intersection = (
                    shape_union.intersection(cell_left).area
                    + shape_union.intersection(cell_right).area
                )
                cell_area = cell_left.area + cell_right.area

            # Sonderfall: Zelle überschreitet den Antimeridian (-180°) nach links
            # → analog: linke Hälfte wird auf die rechte Seite gespiegelt
            elif lon_left < -180:
                cell_left = box(lon_left + 360, lat_lower, 180, lat_upper)
                cell_right = box(-180, lat_lower, lon_right, lat_upper)
                intersection = (
                    shape_union.intersection(cell_left).area
                    + shape_union.intersection(cell_right).area
                )
                cell_area = cell_left.area + cell_right.area

            # Normalfall: Zelle liegt vollständig innerhalb von ±180°
            else:
                cell = box(lon_left, lat_lower, lon_right, lat_upper)
                intersection = shape_union.intersection(cell).area
                cell_area = cell.area

            # Landanteil = Schnittfläche mit Landpolygon / Gesamtfläche der Zelle
            fractions[i, j] = intersection / cell_area

    # Als xarray DataArray zurückgeben mit denselben Koordinaten wie Eingangsarray
    return xr.DataArray(
        fractions,
        coords={"lat": da.lat, "lon": da.lon},
        dims=["lat", "lon"],
        attrs={"long_name": "Land fraction", "units": "1"},
    )
