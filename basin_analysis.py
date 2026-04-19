import xarray as xr
import geopandas as gpd
import numpy as np
import cftime
import matplotlib.pyplot as plt
from src.preprocessing.land_mask import create_mask
from snow_analysis import compute_grid_cell_area

"""plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)"""

# noch die Plots für Abfluss des Einzugsgebiets ergänzen


def convert_mm_month_to_discharge_m3_month(da, river_mask, cell_area):
    snow_melt_m = da / 1000  # mm → m
    return snow_melt_m * cell_area * river_mask


def monthly_discharge_sum_m3(da):
    total_volume_monthly = da.sum(dim=["lat", "lon"])
    return total_volume_monthly


def plot_monthly_discharge(*dataarrays, labels=None, years=0):
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
    import matplotlib.pyplot as plt

    n_months = 180
    months = np.arange(n_months)

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

    fig, ax = plt.subplots(figsize=(8, 5))

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


def months_between(dt1, dt2):
    """Berechnet die Anzahl der Monate zwischen zwei cftime-Objekten."""
    return (dt1.year - dt2.year) * 12 + (dt1.month - dt2.month)


def plot_discharge(*dataarrays, labels=None, years=0):
    import matplotlib.pyplot as plt
    import numpy as np
    import cftime

    # Labels validieren
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

    # Referenzzeitpunkt
    sim_start = cftime.DatetimeNoLeap(5, 2, 1, 0, 0, 0, 0, has_year_zero=True)

    # Zeitscheibe auswählen
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

    n_months = len(arrays_to_plot[0].time)
    months = np.arange(n_months)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Vertikale Linien jeden Monat
    for j in np.arange(0, n_months, 1):
        ax.axvline(j, color="lightgrey", linewidth=0.2)

    # Rote gestrichelte Linie bei Monat 4
    ax.axvline(4, color="red", linewidth=0.5, linestyle="dashed")

    # Dicke Linie bei jedem Januar
    jan_ticks = np.arange(0, n_months, 12)
    for j in jan_ticks:
        ax.axvline(j, color="lightgrey", linewidth=0.5)

    # X-Achse
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Jan / Year {month_offset + i}" for i in range(len(jan_ticks))],
        rotation=45,
        size=5,
        ha="right",
    )

    # Zeitreihen plotten
    for da, label in zip(arrays_to_plot, labels):
        ax.plot(months, da.values, linewidth=1.2, label=label)

    ax.set_title(f"{river} - {year_label} Years")
    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Discharge from Snow Melt (m3/s)")
    ax.legend()
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
    fig, ax = plt.subplots(figsize=(8, 5))

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


def plot_annual_sums_two_bars(
    scenario, control, labels=["Scenario", "Control"], ax=None
):
    n_years = len(scenario)
    year_indices = np.arange(n_years)

    bar_width = 0.4
    offset = bar_width / 2

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        standalone = True
    else:
        standalone = False

    ax.bar(
        year_indices - offset,
        scenario.values,
        width=bar_width,
        color="steelblue",
        alpha=0.85,
        label=labels[0],
    )
    ax.bar(
        year_indices + offset,
        control.values,
        width=bar_width,
        color="grey",
        alpha=0.85,
        label=labels[1],
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(year_indices)
    ax.set_xticklabels(year_indices, fontsize=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Snowmelt [m3]")
    ax.legend(fontsize=8)
    ax.set_title(f"{river}")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_annual_anomaly(annual_scenario_ts, annual_control_ts, ax=None):
    # annual_control_ts kann entweder die gemittelte Referenz (Skalar)
    # oder die vollständige Zeitreihe sein – wir brauchen beides
    # Daher: Referenzwert = Mittelwert, SD aus der vollen Zeitreihe
    control_mean = annual_control_ts.mean(dim="year")
    control_std = annual_control_ts.std(dim="year")

    # SD in Prozent des Mittelwerts
    sd_pct = (control_std / control_mean * 100).values

    # Anomalie in Prozent relativ zum Mittelwert
    anomaly = (annual_scenario_ts - control_mean) / control_mean * 100
    print(f"{river} Max:{anomaly.values.max()}, Min: {anomaly.values.min()}")
    n_years = len(anomaly)
    year_indices = np.arange(n_years)
    values = anomaly.values

    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.06)
        standalone = True
    else:
        standalone = False

    # ±1 SD als gestrichelte Linien
    ax.axhline(
        sd_pct,
        color="grey",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )
    ax.axhline(
        -sd_pct,
        color="grey",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )

    ax.bar(year_indices, pos_vals, color="steelblue", alpha=0.85)
    ax.bar(year_indices, neg_vals, color="steelblue", alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(year_indices)
    ax.set_xticklabels(year_indices, fontsize=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Snowmelt anomaly (%)")
    ax.set_title(f"{river}")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_monthly_bars(scenario, control):
    import matplotlib.pyplot as plt
    import numpy as np

    n_months = len(scenario)
    x = np.arange(n_months)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Vertikale Hilfslinien jeden Monat
    for j in x:
        ax.axvline(j, color="lightgrey", linewidth=0.2)

    # Dickere Linie bei jedem Januar
    jan_ticks = np.arange(0, n_months, 12)
    for j in jan_ticks:
        ax.axvline(j, color="grey", linewidth=0.5)

    # Control nach unten (negiert), Szenario nach oben
    ax.bar(
        x, -control.values, width=1.0, color="steelblue", alpha=0.85, label="Control"
    )
    ax.bar(x, scenario.values, width=1.0, color="tomato", alpha=0.85, label="Szenario")

    ax.axhline(0, color="black", linewidth=0.8)

    # Y-Achse: absolute Werte anzeigen (nicht negiert)
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])

    # X-Achse: Januar-Ticks mit Jahreslabel
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Jan. / Year {i}" for i in range(len(jan_ticks))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Discharge [m³/s]")
    ax.set_title(f"{river}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./results/basin/{river}_monthly_bars.png", dpi=300)
    plt.close(fig)


def plot_monthly_anomaly(scenario, control):
    import matplotlib.pyplot as plt
    import numpy as np

    anomaly = scenario - control

    n_months = len(anomaly)
    month_indices = np.arange(n_months)
    values = anomaly.values

    # Positive und negative Werte trennen
    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Vertikale Hilfslinien jeden Monat
    for j in month_indices:
        ax.axvline(j, color="lightgrey", linewidth=0.2)

    # Dickere Linie bei jedem Januar
    jan_ticks = np.arange(0, n_months, 12)
    for j in jan_ticks:
        ax.axvline(j, color="grey", linewidth=0.5)

    # Control nach unten (negiert), Szenario nach oben
    ax.bar(
        month_indices,
        pos_vals,
        width=1.0,
        color="steelblue",
        alpha=0.85,
        label="Control",
    )
    ax.bar(
        month_indices, neg_vals, width=1.0, color="tomato", alpha=0.85, label="Szenario"
    )

    ax.axhline(0, color="black", linewidth=0.8)

    # Y-Achse: absolute Werte anzeigen (nicht negiert)
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])

    # X-Achse: Januar-Ticks mit Jahreslabel
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Jan / Year {i}" for i in range(len(jan_ticks))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Time (months)")
    ax.set_ylabel("Snowmelt anomaly [m³]")
    ax.set_title(f"{river}")
    plt.tight_layout()
    plt.savefig(f"./results/basin/{river}_monthly_anomaly.png", dpi=300)
    plt.close(fig)


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


""""Rhine": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040023010,
        ],"""

""""Jennisei": [
            "./data/HydroBasins/hybas_si_lev01-12_v1c/hybas_si_lev04_v1c.shp",
            3040004740,
        ],"""

""""Nile": [
            "./data/HydroBasins/hybas_af_lev01-12_v1c/hybas_af_lev04_v1c.shp",
            1040034260,
        ],"""
""""Mekong": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040017020,
        ]"""

""""Indus": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040033640,
        ],  # sieht anders aus als im Internet, aber laut HydroBasins ist das der Indus"""
"""        "Jangtsekiang": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040009880,
        ],  # sieht anders aus als im Internet, aber laut HydroBasins ist das der Jangtsekiang"""

if __name__ == "__main__":
    main_rivers = {
        "Donau": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040008490,
        ],
        "Ganges - Brahmaputra - Meghna?": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040025450,
        ],
        "Yellow River": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040007850,
        ],
        "Tigris - Euphrates": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040073570,
        ],
        "Ob": [
            "./data/HydroBasins/hybas_si_lev01-12_v1c/hybas_si_lev04_v1c.shp",
            3040001840,
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

    cell_area = compute_grid_cell_area(ds_control.snow_storage)  # km2

    annual_anomaly_data = (
        {}
    )  # {river_name: (dsc_45_annual_sum, dsc_control_annual_sum)}

    for river, (filepath, river_id) in main_rivers.items():
        # Select river from Data
        basins_lev4 = gpd.read_file(filepath)
        river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
        river_basin_diss = river_basin.dissolve()
        # Fraction of Raster Cells in River Basin
        river_mask = create_mask(ds_150, river_basin_diss)

        # Select all cells inside Basin
        river_150 = ds_150.snow_melt.where(river_mask > 0)
        river_control = ds_control.snow_melt.where(river_mask > 0)
        river_16 = ds_16.snow_melt.where(river_mask > 0)
        river_47 = ds_47.snow_melt.where(river_mask > 0)

        # Snow Storage in River Basin
        river_47_storage = ds_47.snow_storage.where(river_mask > 0)
        river_control_storage = ds_control.snow_storage.where(river_mask > 0)

        # convert from mm/month to discharge in m3/month
        dsc_150 = convert_mm_month_to_discharge_m3_month(
            river_150, river_mask, cell_area
        )
        dsc_control = convert_mm_month_to_discharge_m3_month(
            river_control, river_mask, cell_area
        )
        dsc_47 = convert_mm_month_to_discharge_m3_month(river_47, river_mask, cell_area)
        dsc_16 = convert_mm_month_to_discharge_m3_month(river_16, river_mask, cell_area)
        storage_47 = convert_mm_month_to_discharge_m3_month(
            river_47_storage, river_mask, cell_area
        )
        storage_control = convert_mm_month_to_discharge_m3_month(
            river_control_storage, river_mask, cell_area
        )

        # Monthly Discharge Sum in m3/month
        dsc_sum_150 = monthly_discharge_sum_m3(dsc_150)
        dsc_sum_47 = monthly_discharge_sum_m3(dsc_47)
        dsc_sum_16 = monthly_discharge_sum_m3(dsc_16)
        dsc_sum_control = monthly_discharge_sum_m3(dsc_control)

        # Monthly Discharge in m3/s
        dsc_150_m3_s = dsc_sum_150 / (
            ds_150["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )
        dsc_47_m3_s = dsc_sum_47 / (
            ds_47["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )
        dsc_16_m3_s = dsc_sum_16 / (
            ds_16["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )
        dsc_control_m3_s = dsc_sum_control / (
            ds_control["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )

        # Monthly Snow Storage Sum in m3/month
        storage_sum_47_m3 = monthly_discharge_sum_m3(storage_47)
        storage_sum_control_m3 = monthly_discharge_sum_m3(storage_control)

        # Annual Discharge Sum in m3/year
        dsc_47_annual_sum = annual_sum(dsc_sum_47)
        dsc_control_annual_sum = annual_sum(dsc_sum_control)

        # Mean Annual Sum in m3/year -> um Schwankung zu reduzieren
        dsc_control_mean_annual_sum = dsc_control_annual_sum.mean(dim="year")

        # Annual Anomaly in m3/year
        annual_anomaly_data[river] = (dsc_47_annual_sum, dsc_control_annual_sum)

        """plot_discharge(
            dsc_sum_47,
            dsc_sum_control,
            dsc_sum_47_ss,
            dsc_sum_control_ss,
            labels=["47", "Control", "47 Snow Storage", "Control Snow Storage"],
            years=[0, 7],
        )
        plot_discharge(
            dsc_sum_47,
            dsc_sum_control,
            dsc_sum_47_ss,
            dsc_sum_control_ss,
            labels=["47", "Control", "47 Snow Storage", "Control Snow Storage"],
            years=[0, 15],
        )
        plot_annual_anomaly(dsc_45_annual_sum, dsc_control_annual_sum)
        plot_annual_anomaly(dsc_45_annual_sum, dsc_control_annual_sum.mean(dim="year"))"""
        print(
            f"{river} - Szenario 32k: {np.sum(dsc_47_m3_s.values > 32000)}, Control 32k: {np.sum(dsc_control_m3_s.values > 32000)} \n Szenario 25k: {np.sum(dsc_47_m3_s.values > 25000)}, Control 25k: {np.sum(dsc_control_m3_s.values > 25000)}"
        )
        print(
            f"{river} - Szenario 32k: {np.sum(dsc_47_m3_s.values > 30000)}, Control 32k: {np.sum(dsc_control_m3_s.values > 30000)}"
        )

        print(
            dsc_47_m3_s[dsc_47_m3_s.values > 32000],
            dsc_control_m3_s[dsc_control_m3_s.values > 32000],
        )
        plot_monthly_anomaly(dsc_sum_47, dsc_sum_control)
        # plot_monthly_bars(dsc_sum_47, dsc_sum_control)
        plot_monthly_bars(dsc_47_m3_s, dsc_control_m3_s)

    n_rivers = len(annual_anomaly_data)
    ncols = 3
    nrows = (n_rivers + ncols - 1) // ncols  # = 4 bei 12 Flüssen

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6, nrows * 5),  # Höhe pro Zeile erhöht
        constrained_layout=True,  # statt tight_layout
    )
    axes_flat = axes.flatten()

    for ax, (river, (scenario, control)) in zip(axes_flat, annual_anomaly_data.items()):
        plot_annual_anomaly(scenario, control, ax=ax)

    # Übrige leere Subplots ausblenden
    for ax in axes_flat[n_rivers:]:
        ax.set_visible(False)

    # fig.suptitle("Jährliche Anomalie der Abflussspende", fontsize=11)
    plt.tight_layout()
    plt.savefig("./results/river_basin_annual_anomalies.png", dpi=300)
    plt.close(fig)

    fig2, axes2 = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6, nrows * 5),
        constrained_layout=True,
    )
    axes_flat2 = axes2.flatten()

    for ax, (river, (scenario, control)) in zip(
        axes_flat2, annual_anomaly_data.items()
    ):
        plot_annual_sums_two_bars(scenario, control, ax=ax)

    for ax in axes_flat2[n_rivers:]:
        ax.set_visible(False)

    plt.savefig("./results/river_basin_annual_sums.png", dpi=300)
    plt.close(fig2)
