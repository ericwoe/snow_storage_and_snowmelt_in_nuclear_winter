import xarray as xr
import geopandas as gpd
import numpy as np
import cftime
import matplotlib.pyplot as plt

# noch die Plots für Abfluss des Einzugsgebiets ergänzen


def create_mask(ds: xr.Dataset, shape: gpd.GeoDataFrame) -> xr.DataArray:
    """
    Create a fractional land mask from a GADM shapefile for the lat/lon grid of a dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'lat' and 'lon' coordinates
    shape : Geodataframe
        Geodataframe with shapes/polygons

    Returns
    -------
    land_mask_xr : xr.DataArray
        Fractional land mask (0.0-1.0 representing fraction of cell inside shape)
        with same lat/lon as dataset.
        Longitude is returned in 0-360 range for einfache Verwendung.
    """
    from shapely.geometry import box

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

    # Merge all shapes into one polygon (falls mehrere Polygone im GeoDataFrame)
    shape_union = shape.union_all()

    # Calculate fractional coverage for each cell
    fractional_coverage = np.zeros(len(polygons), dtype=float)

    for i, cell in enumerate(cells_gdf.geometry):
        if cell.intersects(shape_union):
            intersection = cell.intersection(shape_union)
            fractional_coverage[i] = intersection.area / cell.area

    # Reshape to 2D grid
    land_mask = fractional_coverage.reshape(len(lat), len(lon))

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

    return land_mask_xr


def convert_mm_month_to_discharge_m3_month(da):
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

    ds_150 = xr.open_dataset("./results/snow_dataset_150_tg.nc")
    ds_control = xr.open_dataset("./results/snow_dataset_nw_cntrl_03.nc")

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

        plot_weighted_monthly_mean(river_150, river_control, years=0)
        plot_weighted_monthly_mean(river_150, river_control, years=10)
        plot_weighted_monthly_mean(river_150, river_control, years=5)

        plot_annual_sum(river_150, river_control)

        dsc_150 = convert_mm_month_to_discharge_m3_month(river_150)
        dsc_control = convert_mm_month_to_discharge_m3_month(river_control)

        dsc_sum_150 = monthly_discharge_sum(dsc_150)
        dsc_sum_control = monthly_discharge_sum(dsc_control)

        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=0)
        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=10)
        plot_monthly_discharge(dsc_sum_150, dsc_sum_control, years=5)
