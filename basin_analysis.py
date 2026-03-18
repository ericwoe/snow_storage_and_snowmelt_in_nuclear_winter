import xarray as xr
import geopandas as gpd
import numpy as np
import cftime
import matplotlib.pyplot as plt
from src.preprocessing.land_mask import create_mask

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)

# noch die Plots für Abfluss des Einzugsgebiets ergänzen


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
    fig, ax = plt.subplots(figsize=(16, 5))

    # Beide Zeitreihen in dieselbe Achse plotten
    ts_150_plot.plot(ax=ax, label="47 Tg Szenario")
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


def months_between(dt1, dt2):
    """Berechnet die Anzahl der Monate zwischen zwei cftime-Objekten."""
    return (dt1.year - dt2.year) * 12 + (dt1.month - dt2.month)


def plot_discharge(ts_scenario, ts_control):
    import matplotlib.pyplot as plt

    n_months = 180
    months = np.arange(n_months)

    fig, ax = plt.subplots(figsize=(18, 4))

    # Vertikale Linie bei JEDEM Monat über alle Jahre
    for j in np.arange(0, n_months, 1):
        ax.axvline(j, color="lightgrey", linewidth=0.2)
    ax.axvline(4, color="red", linewidth=0.5, linestyle="dashed")
    # Dicke Linie + Beschriftung bei jedem Januar
    jan_ticks = np.arange(0, n_months, 12)
    for i, j in enumerate(jan_ticks):
        ax.axvline(j, color="lightgrey", linewidth=0.5)

    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Jan / Year {i}" for i in range(15)], rotation=45, size=5, ha="right"
    )
    ax.plot(months, ts_scenario, linewidth=1.2, label="47 Tg Szenario")
    ax.plot(months, ts_control, linewidth=1.2, label="Control Szenario")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Discharge from Snow Melt (m3/s)")
    ax.legend()


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
        labels = ["Serie " + str(i + 1) for i in range(len(dataarrays))]
    elif len(labels) != len(dataarrays):
        raise ValueError(
            "Anzahl der Labels ("
            + str(len(labels))
            + ") muss mit Anzahl der DataArrays ("
            + str(len(dataarrays))
            + ") übereinstimmen."
        )

    # Referenzzeitpunkt: Beginn der Simulation
    sim_start = cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True)

    # Zeitscheibe auswählen, falls years angegeben
    if years:
        start = cftime.DatetimeNoLeap(
            5 + years[0], 2, 1, 0, 0, 0, 0, has_year_zero=True
        )
        end = cftime.DatetimeNoLeap(5 + years[1], 2, 1, 0, 0, 0, 0, has_year_zero=True)
        time_slice = slice(start, end)
        arrays_to_plot = [da.sel(time=time_slice) for da in dataarrays]
        year_label = str(years[1] - years[0])
        month_offset = months_between(start, sim_start)
    else:
        arrays_to_plot = list(dataarrays)
        year_label = "All"
        month_offset = 0

    # Figur und Achse erzeugen
    fig, ax = plt.subplots(figsize=(16, 5))

    # Alle Zeitreihen plotten
    for da, label in zip(arrays_to_plot, labels):
        da.plot(ax=ax, label=label)

    # Ticks in 12er-Schritten mit dynamischem Offset
    n_months = len(arrays_to_plot[0].time)
    tick_positions = list(range(1, n_months, 12))
    tick_labels = [str(month_offset + i) for i in tick_positions]
    ax.set_xticks([arrays_to_plot[0].time.values[i] for i in tick_positions])
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.tick_params(axis="x", which="minor", bottom=False)

    # Achsenbeschriftung
    ax.set_title(f"{river} - {year_label} Years")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Discharge from Snow Melt (m3/s)")

    ax.legend()
    plt.tight_layout()
    plt.show()
    print(arrays_to_plot)


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

        plot_discharge(dsc_sum_47, dsc_sum_control)

        plot_monthly_discharge(dsc_sum_47, dsc_sum_control, years=0)
        plot_monthly_discharge(dsc_sum_47, dsc_sum_control, years=3)
        plot_monthly_discharge(dsc_sum_47, dsc_sum_control, years=5)
        plot_monthly_discharge(dsc_sum_47, dsc_sum_control, years=10)

        plot_monthly_discharge_args(
            dsc_sum_47,
            dsc_sum_control,
            labels=["47 Tg", "Control"],
            years=[0, 10],
        )

        plot_monthly_discharge_args(
            dsc_sum_47,
            dsc_sum_control,
            labels=["47 Tg", "Control"],
            years=[0, 5],
        )
        plot_monthly_discharge_args(
            dsc_sum_47,
            dsc_sum_control,
            labels=["47 Tg", "Control"],
            years=[5, 10],
        )
        plot_monthly_discharge_args(
            dsc_sum_47,
            dsc_sum_control,
            labels=["47 Tg", "Control"],
            years=[10, 15],
        )

        print(dsc_sum_150.where(dsc_sum_150 > 9500, drop=True))
        print(dsc_sum_control.where(dsc_sum_control > 9500, drop=True))
