import xarray as xr
import geopandas as gpd
import numpy as np
import cftime
import matplotlib.pyplot as plt

# noch die Plots für Abfluss des Einzugsgebiets ergänzen


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


def convert_mm_month_to_discharge_m3_month(da, river_mask):
    R = 6371000  # Earthradius in m
    lat_rad = np.deg2rad(da.lat.values)
    lon_rad = np.deg2rad(da.lon.values)

    # Gitterabstände in Radiant
    dlat = np.abs(lat_rad[1] - lat_rad[0])
    dlon = np.abs(lon_rad[1] - lon_rad[0])

    # Zellflächen (lat, lon) → 2D Array
    cell_area = (R**2) * dlat * dlon * np.cos(lat_rad)[:, None]  # Shape (96,144)

    snow_melt_m = da / 1000  # mm → m

    return snow_melt_m * cell_area * river_mask


def monthly_discharge_sum(da):
    total_volume_monthly = da.sum(dim=["lat", "lon"])
    seconds_in_month = 30 * 24 * 3600  # grob, alternativ kalendarisch je Monat
    return total_volume_monthly / seconds_in_month


def plot_monthly_discharge(da_ts_150, da_ts_ctrl, years=0):

    if years:
        ts_150_plot = da_ts_150.sel(
            time=slice(
                cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True),
                cftime.DatetimeNoLeap(5 + years, 2, 1, 0, 0, 0, 0, has_year_zero=True),
            )
        )
        ts_ctrl_plot = da_ts_ctrl.sel(
            time=slice(
                cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True),
                cftime.DatetimeNoLeap(5 + years, 2, 1, 0, 0, 0, 0, has_year_zero=True),
            )
        )
    else:
        ts_ctrl_plot = da_ts_ctrl
        ts_150_plot = da_ts_150
        years = "All"

    # Neue Figur und Achse erzeugen
    fig, ax = plt.subplots(figsize=(8, 5))

    # Beide Zeitreihen in dieselbe Achse plotten
    ts_150_plot.plot(ax=ax, label="150 Tg Szenario")
    ts_ctrl_plot.plot(ax=ax, label="Control Szenario")

    # Achsenbeschriftung
    ax.set_title(f"{river} - {years} Years")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Discharge from Snow Melt (m3/s)")

    # Legende hinzufügen
    ax.legend()

    # Layout verbessern
    plt.tight_layout()
    plt.show()


def plot_monthly_discharge_args(*dataarrays, labels=None, years=None):
    """
    Plottet monatliche Abflusszeitreihen für beliebig viele xarray DataArrays.

    Parameter:
    ----------
    *dataarrays : xarray.DataArray
        Beliebig viele DataArrays mit Zeitdimension.
    labels : list[str], optional
        Beschriftungen für die Legende. Falls None, werden "Serie 1", "Serie 2", ... verwendet.
    years : list of int, optional
        Anzahl der Jahre ab Jahr 5. Bei 0 werden alle Daten geplottet.
    """
    if labels is None:
        labels = [f"Serie {i+1}" for i in range(len(dataarrays))]
    elif len(labels) != len(dataarrays):
        raise ValueError(
            f"Anzahl der Labels ({len(labels)}) muss mit Anzahl der DataArrays ({len(dataarrays)}) übereinstimmen."
        )

    # Zeitscheibe auswählen, falls years angegeben
    if years:
        time_slice = slice(
            cftime.DatetimeNoLeap(5 + years[0], 2, 1, 0, 0, 0, 0, has_year_zero=True),
            cftime.DatetimeNoLeap(5 + years[1], 2, 1, 0, 0, 0, 0, has_year_zero=True),
        )
        arrays_to_plot = [da.sel(time=time_slice) for da in dataarrays]
        year_label = years[1] - years[0]
    else:
        arrays_to_plot = list(dataarrays)
        year_label = "All"

    # Figur und Achse erzeugen
    fig, ax = plt.subplots(figsize=(8, 5))

    # Alle Zeitreihen plotten
    for da, label in zip(arrays_to_plot, labels):
        da.plot(ax=ax, label=label)

    # Achsenbeschriftung
    ax.set_title(f"{river} - {year_label} Years")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Discharge from Snow Melt (m3/s)")

    ax.legend()
    plt.tight_layout()
    plt.show()


def annual_sum(da):
    # Zeitkoordinate um einen Monat verschieben
    da_new_time = da.assign_coords(
        time=[
            cftime.DatetimeNoLeap(
                t.year if t.month > 1 else t.year - 1,
                t.month - 1 if t.month > 1 else 12,
                1,
            )
            for t in da.time.values
        ]
    )
    return da_new_time.groupby("time.year").sum("time")


def weighted_monthly_mean(da, mask):
    return (da * mask).sum(dim=["lat", "lon"]) / mask.sum(dim=["lat", "lon"])


def plot_annual_sum(da_150, da_control):
    monthly_150 = weighted_monthly_mean(da_150, river_mask)
    monthly_control = weighted_monthly_mean(da_control, river_mask)

    annual_150 = annual_sum(monthly_150)
    annual_control = annual_sum(monthly_control)

    # Neue Figur und Achse erzeugen
    fig, ax = plt.subplots(figsize=(8, 5))

    # Beide Zeitreihen in dieselbe Achse plotten
    annual_150.plot(ax=ax, label="150 Tg Szenario")
    annual_control.plot(ax=ax, label="Control Szenario")

    # Achsenbeschriftung (optional, aber empfehlenswert)
    ax.set_title(river)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Snow_Melt (mm/year)")

    # Legende hinzufügen
    ax.legend()

    # Layout verbessern
    plt.tight_layout()
    plt.show()


def plot_weighted_monthly_mean(da_river_150, da_river_ctrl, years=0):
    # Daten für das Plotten vorbereiten
    ts_150_plot = weighted_monthly_mean(da_river_150, river_mask)
    ts_ctrl_plot = weighted_monthly_mean(da_river_ctrl, river_mask)

    if years:
        ts_150_plot = ts_150_plot.sel(
            time=slice(
                cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True),
                cftime.DatetimeNoLeap(5 + years, 2, 1, 0, 0, 0, 0, has_year_zero=True),
            )
        )
        ts_ctrl_plot = ts_ctrl_plot.sel(
            time=slice(
                cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True),
                cftime.DatetimeNoLeap(5 + years, 2, 1, 0, 0, 0, 0, has_year_zero=True),
            )
        )
    else:
        years = "All"

    # Neue Figur und Achse erzeugen
    fig, ax = plt.subplots(figsize=(8, 5))

    # Beide Zeitreihen in dieselbe Achse plotten
    ts_150_plot.plot(ax=ax, label="150 Tg Szenario")
    ts_ctrl_plot.plot(ax=ax, label="Control Szenario")

    # Achsenbeschriftung
    ax.set_title(f"{river} - {years} Years")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Snow_Melt (mm/month)")

    # Legende hinzufügen
    ax.legend()

    # Layout verbessern
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_rivers = {
        "Rhine": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040023010,
        ],
        "Yellow River": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040007850,
        ],
        "Nile": [
            "./data/HydroBasins/hybas_af_lev01-12_v1c/hybas_af_lev04_v1c.shp",
            1040034260,
        ],
        "Mississippi": [
            "./data/HydroBasins/hybas_na_lev01-12_v1c/hybas_na_lev04_v1c.shp",
            7040047060,
        ],
    }

    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_control = xr.open_dataset("./results/Control/snow_control.nc")

    for river, (filepath, river_id) in main_rivers.items():
        # Select river from Data
        basins_lev4 = gpd.read_file(filepath)
        river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
        river_basin_diss = river_basin.dissolve()
        # Fraction of Raster Cells in River Basin
        river_mask = create_mask(ds_150, river_basin_diss)
        # All cells inside Basin
        river_150 = ds_150.snow_melt.where(river_mask > 0)
        river_control = ds_control.snow_melt.where(river_mask > 0)
        river_16 = ds_16.snow_melt.where(river_mask > 0)
        river_47 = ds_47.snow_melt.where(river_mask > 0)

        """plot_weighted_monthly_mean(river_150, river_control, years=0)
        plot_weighted_monthly_mean(river_150, river_control, years=10)
        plot_weighted_monthly_mean(river_150, river_control, years=5)

        plot_annual_sum(river_150, river_control)"""

        dsc_150 = convert_mm_month_to_discharge_m3_month(river_150, river_mask)
        dsc_control = convert_mm_month_to_discharge_m3_month(river_control, river_mask)
        dsc_47 = convert_mm_month_to_discharge_m3_month(river_47, river_mask)
        dsc_16 = convert_mm_month_to_discharge_m3_month(river_16, river_mask)

        dsc_sum_150 = monthly_discharge_sum(dsc_150)
        dsc_sum_47 = monthly_discharge_sum(dsc_47)
        dsc_sum_16 = monthly_discharge_sum(dsc_16)
        dsc_sum_control = monthly_discharge_sum(dsc_control)

        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=0)
        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=3)
        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=5)
        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=10)

        plot_monthly_discharge_args(
            dsc_sum_150,
            dsc_sum_47,
            dsc_sum_16,
            dsc_sum_control,
            labels=["150 Tg", "47 Tg", "16 Tg", "Control"],
            years=[0, 10],
        )

        plot_monthly_discharge_args(
            dsc_sum_150,
            dsc_sum_47,
            dsc_sum_16,
            dsc_sum_control,
            labels=["150 Tg", "47 Tg", "16 Tg", "Control"],
            years=[0, 5],
        )
        plot_monthly_discharge_args(
            dsc_sum_150,
            dsc_sum_47,
            dsc_sum_16,
            dsc_sum_control,
            labels=["150 Tg", "47 Tg", "16 Tg", "Control"],
            years=[5, 10],
        )
        plot_monthly_discharge_args(
            dsc_sum_150,
            dsc_sum_47,
            dsc_sum_16,
            dsc_sum_control,
            labels=["150 Tg", "47 Tg", "16 Tg", "Control"],
            years=[10, 15],
        )

        print(dsc_sum_150.where(dsc_sum_150 > 9500, drop=True))
        print(dsc_sum_control.where(dsc_sum_control > 9500, drop=True))
