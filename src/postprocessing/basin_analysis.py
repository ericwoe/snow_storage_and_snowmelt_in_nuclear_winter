import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import xarray as xr
import geopandas as gpd
import numpy as np
import cftime
import matplotlib.pyplot as plt
from src.preprocessing.land_mask import create_mask
from src.postprocessing.snow_analysis import compute_grid_cell_area

plt.style.use(
    "https://raw.githubusercontent.com/allfed/ALLFED-matplotlib-style-sheet/main/ALLFED.mplstyle"
)

# noch die Plots für Abfluss des Einzugsgebiets ergänzen


def convert_mm_month_to_discharge_m3_month(da, river_mask, cell_area):
    snow_melt_m = da / 1000  # mm → m
    return snow_melt_m * cell_area * river_mask


def monthly_discharge_sum_m3(da):
    total_volume_monthly = da.sum(dim=["lat", "lon"])
    return total_volume_monthly


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

    control_mean = control.mean(dim="year")
    control_std = control.std(dim="year")

    ax.bar(
        year_indices - offset,
        scenario.values,
        width=bar_width,
        color="steelblue",
        alpha=0.8,
        label=labels[0],
    )
    ax.bar(
        year_indices + offset,
        control_mean.values,
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


def plot_annual_anomaly(annual_scenario_ts, annual_control_ts, ax=None, title=None):
    # annual_control_ts kann entweder die gemittelte Referenz (Skalar)
    # oder die vollständige Zeitreihe sein – wir brauchen beides
    # Daher: Referenzwert = Mittelwert, SD aus der vollen Zeitreihe
    control_mean = annual_control_ts.mean(dim="year")
    control_std = annual_control_ts.std(dim="year")

    # SD in Prozent des Mittelwerts
    sd_pct = (control_std / control_mean * 100).values

    # Anomalie in Prozent relativ zum Mittelwert
    anomaly = (annual_scenario_ts - control_mean) / control_mean * 100
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
        color="#E34444",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )
    ax.axhline(
        -sd_pct,
        color="#E34444",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )

    ax.bar(year_indices, pos_vals, color="#4575b4", alpha=0.8)
    ax.bar(year_indices, neg_vals, color="#4575b4", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(year_indices)
    ax.set_xticklabels(year_indices, fontsize=5)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel(f"{title} Anomaly (%)")
    ax.set_title(f"{river}")

    if standalone:
        plt.tight_layout()
        plt.savefig(f"./results/basin/{river}_annual_{title}_sum_anomaly.png", dpi=300)
        plt.close(fig)


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
    ax.bar(x, -control.values, width=1.0, color="red", alpha=0.8, label="Control")
    ax.bar(x, scenario.values, width=1.0, color="#3A6A91", alpha=0.8, label="Szenario")

    ax.axhline(0, color="black", linewidth=0.8)

    # Y-Achse: absolute Werte anzeigen (nicht negiert)
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])

    # X-Achse: Januar-Ticks mit Jahreslabel
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Year {i}" for i in range(len(jan_ticks))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Time [months]")
    ax.set_ylabel("Discharge [m³/s]")
    ax.set_title(f"{river}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./results/basin/{river}_monthly_bars.png", dpi=300)
    plt.close(fig)


def plot_monthly_bars_rain_snow(
    scenario_snowmelt, scenario_rain, control_snowmelt, control_rain, river, ax=None
):
    import matplotlib.pyplot as plt
    import numpy as np

    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        standalone = True

    n_months = len(scenario_snowmelt)
    x = np.arange(n_months)

    for j in x:
        ax.axvline(j, color="lightgrey", linewidth=0.2)
    jan_ticks = np.arange(0, n_months, 12)
    for j in jan_ticks:
        ax.axvline(j, color="grey", linewidth=0.5)

    color_control = "#E34444"
    color_scenario = "#4575b4"
    print(f"{river}")
    print(f"Maximum Snowmelt:{scenario_snowmelt.values.max()}")
    print(
        f"5 biggest Months Control: {(control_snowmelt+control_rain).sortby((control_snowmelt+control_rain), ascending=False)[:5]}"
    )
    print(
        f"10 biggest Months Scenario: {(scenario_snowmelt+scenario_rain).sortby((scenario_snowmelt+scenario_rain), ascending=False)[:10]}"
    )

    ax.bar(
        x,
        scenario_rain.values,
        width=1.0,
        color=color_scenario,
        alpha=0.8,
        label="Scenario Rain",
    )
    ax.bar(
        x,
        scenario_snowmelt.values,
        bottom=scenario_rain.values,
        width=1.0,
        color=color_scenario,
        alpha=0.4,
        label="Scenario Snowmelt",
    )
    ax.bar(
        x,
        -control_rain.values,
        width=1.0,
        color=color_control,
        alpha=0.8,
        label="Control Rain",
    )
    ax.bar(
        x,
        -control_snowmelt.values,
        bottom=-control_rain.values,
        width=1.0,
        color=color_control,
        alpha=0.4,
        label="Control Snowmelt",
    )

    ax.axhline(0, color="black", linewidth=0.8)
    yticks = ax.get_yticks()
    ax.set_yticklabels([str(abs(int(y))) for y in yticks])
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Year {i}" for i in range(len(jan_ticks))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Time [months]", fontsize=9)
    ax.set_ylabel("Discharge [m³/s]", fontsize=10)
    ax.set_title(river, fontsize=13)

    if ax is ax.get_figure().axes[0]:
        ax.legend(fontsize=6)

    if standalone:
        plt.tight_layout()
        plt.savefig(f"./results/basin/{river}_monthly_bars_rain_snow.png", dpi=300)
        plt.close()


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
        [f"Year {i}" for i in range(len(jan_ticks))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)

    ax.set_xlabel("Time [months]")
    ax.set_ylabel("Discharge anomaly [m³/s]")
    ax.set_title(f"{river}")
    plt.tight_layout()
    plt.savefig(f"./results/basin/{river}_monthly_anomaly.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main_rivers = {
        "Rhine": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040023010,
        ],
        "Ganges-Brahmaputra-Meghna": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040025450,
        ],
        "Yellow River": [
            "./data/HydroBasins/hybas_as_lev01-12_v1c/hybas_as_lev04_v1c.shp",
            4040007850,
        ],
        "Tigris-Euphrates": [
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

    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_control = xr.open_dataset("./results/Control/snow_control.nc")

    cell_area = compute_grid_cell_area(ds_control.snow_storage)  # km2

    annual_snowmelt_anomaly_data = (
        {}
    )  # {river_name: (dsc_45_annual_sum, dsc_control_annual_sum)}
    annual_total_discharge_anomaly_data = {}  # {river_name: (dsc_45_snowmelt_annual_sum

    annual_total_precipitation_anomaly_data = (
        {}
    )  # {river_name: (dsc_45_precip_annual_sum, dsc_control_precip_annual_sum)}

    monthly_snow_rain_data = {}

    for river, (filepath, river_id) in main_rivers.items():
        # Select river from Data
        basins_lev4 = gpd.read_file(filepath)
        river_basin = basins_lev4[basins_lev4["MAIN_BAS"] == river_id]
        river_basin_diss = river_basin.dissolve()
        # Fraction of Raster Cells in River Basin
        river_mask = create_mask(ds_control, river_basin_diss)

        # 1. Select all cells inside Basin
        ##SNOWMELT

        snowmelt_control = ds_control.snow_melt.where(river_mask > 0)

        snowmelt_47 = ds_47.snow_melt.where(river_mask > 0)

        ##RAIN
        rain_47 = ds_47.rainfall.where(river_mask > 0)
        rain_control = ds_control.rainfall.where(river_mask > 0)

        ##PRECIP
        precip_47 = ds_47.precip_mm_month.where(river_mask > 0)
        precip_control = ds_control.precip_mm_month.where(river_mask > 0)

        ##SNOWSTORAGE

        snow_storage_control = ds_control.snow_storage.where(river_mask > 0)
        snow_storage_47 = ds_47.snow_storage.where(river_mask > 0)

        weights = cell_area * river_mask
        mean_snow_control = (ds_control.snow_storage * weights).sum(
            dim=["lat", "lon"]
        ) / weights.sum(dim=["lat", "lon"])
        mean_snow_47 = (ds_47.snow_storage * weights).sum(
            dim=["lat", "lon"]
        ) / weights.sum(dim=["lat", "lon"])
        # 2. convert from mm/month to discharge in m3/month
        ##SNOWMELT
        dsc_control_snowmelt = convert_mm_month_to_discharge_m3_month(
            snowmelt_control, river_mask, cell_area
        )
        dsc_47_snowmelt = convert_mm_month_to_discharge_m3_month(
            snowmelt_47, river_mask, cell_area
        )

        ##RAIN
        dsc_47_rain = convert_mm_month_to_discharge_m3_month(
            rain_47, river_mask, cell_area
        )
        dsc_control_rain = convert_mm_month_to_discharge_m3_month(
            rain_control, river_mask, cell_area
        )

        ##PRECIP
        dsc_control_precip = convert_mm_month_to_discharge_m3_month(
            precip_control, river_mask, cell_area
        )
        dsc_47_precip = convert_mm_month_to_discharge_m3_month(
            precip_47, river_mask, cell_area
        )

        # 3. Monthly Discharge Sum in m3/month
        ##SNOWMELT

        dsc_sum_47_snowmelt = monthly_discharge_sum_m3(dsc_47_snowmelt)
        dsc_sum_control_snowmelt = monthly_discharge_sum_m3(dsc_control_snowmelt)

        ##RAIN
        dsc_sum_47_rain = monthly_discharge_sum_m3(dsc_47_rain)
        dsc_sum_control_rain = monthly_discharge_sum_m3(dsc_control_rain)

        ##PRECIP
        dsc_sum_47_precip = monthly_discharge_sum_m3(dsc_47_precip)
        dsc_sum_control_precip = monthly_discharge_sum_m3(dsc_control_precip)

        # 4. Annual Sum in m3/year
        ##SNOWMELT
        dsc_47_snowmelt_annual_sum = annual_sum(dsc_sum_47_snowmelt)
        dsc_control_snowmelt_annual_sum = annual_sum(dsc_sum_control_snowmelt)

        ##RAIN
        dsc_47_rain_annual_sum = annual_sum(dsc_sum_47_rain)
        dsc_control_rain_annual_sum = annual_sum(dsc_sum_control_rain)

        ##PRECIP
        dsc_47_precip_annual_sum = annual_sum(dsc_sum_47_precip)
        dsc_control_precip_annual_sum = annual_sum(dsc_sum_control_precip)

        # 5. Plotting Annual Anomalies Rain, Snowmelt, Total Discharge
        plot_annual_anomaly(
            dsc_47_snowmelt_annual_sum,
            dsc_control_snowmelt_annual_sum,
            title="Snowmelt",
        )
        plot_annual_anomaly(
            dsc_47_rain_annual_sum, dsc_control_rain_annual_sum, title="Rain"
        )
        plot_annual_anomaly(
            dsc_47_snowmelt_annual_sum + dsc_47_rain_annual_sum,
            dsc_control_snowmelt_annual_sum + dsc_control_rain_annual_sum,
            title="Total Discharge",
        )

        plot_annual_anomaly(
            dsc_47_precip_annual_sum,
            dsc_control_precip_annual_sum,
            title="Precipitation",
        )

        # 6. Save Anomalies in dict for later plotting of all rivers together
        # Annual Anomaly in m3/year
        annual_total_discharge_anomaly_data[river] = (
            dsc_47_snowmelt_annual_sum + dsc_47_rain_annual_sum,
            dsc_control_snowmelt_annual_sum + dsc_control_rain_annual_sum,
        )

        annual_total_precipitation_anomaly_data[river] = (
            dsc_47_precip_annual_sum,
            dsc_control_precip_annual_sum,
        )

        # 7. Monthly Discharge in m3/s
        ##SNOWMELT
        dsc_47_m3_s = dsc_sum_47_snowmelt / (
            ds_47["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )
        dsc_control_m3_s = dsc_sum_control_snowmelt / (
            ds_control["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )
        ##RAIN
        dsc_47_rain_m3_s = dsc_sum_47_rain / (
            ds_47["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )
        dsc_control_rain_m3_s = dsc_sum_control_rain / (
            ds_control["days_in_month"].isel(lat=0, lon=0) * 24 * 3600
        )

        # Mean Annual Sum in m3/year -> um Schwankung zu reduzieren
        ##SNOWMELT
        dsc_control_mean_annual_snowmelt_sum = dsc_control_snowmelt_annual_sum.mean(
            dim="year"
        )
        ##RAIN + SNOWMELT
        dsc_control_mean_annual_sum = (
            dsc_control_rain_annual_sum + dsc_control_snowmelt_annual_sum
        ).mean(dim="year")

        annual_snowmelt_anomaly_data[river] = (
            dsc_47_snowmelt_annual_sum,
            dsc_control_snowmelt_annual_sum,
        )

        monthly_snow_rain_data[river] = {
            "scenario_snowmelt": dsc_47_m3_s,
            "scenario_rain": dsc_47_rain_m3_s,
            "control_snowmelt": dsc_control_m3_s,
            "control_rain": dsc_control_rain_m3_s,
        }

        plot_monthly_bars_rain_snow(
            dsc_47_m3_s,
            dsc_47_rain_m3_s,
            dsc_control_m3_s,
            dsc_control_rain_m3_s,
            river,
        )
        plot_monthly_bars(dsc_47_m3_s, dsc_control_m3_s)

    n_rivers = len(annual_total_discharge_anomaly_data)
    ncols = 3
    nrows = (n_rivers + ncols - 1) // ncols  # = 4 bei 12 Flüssen

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6, nrows * 5),  # Höhe pro Zeile erhöht
        constrained_layout=True,  # statt tight_layout
    )
    axes_flat1 = axes.flatten()

    for ax, (river, (scenario, control)), idx in zip(
        axes_flat1,
        annual_total_discharge_anomaly_data.items(),
        range(len(annual_total_discharge_anomaly_data)),
    ):
        plot_annual_anomaly(scenario, control, ax=ax, title="Annual Total Discharge")
        ax.text(
            -0.10,
            1.07,
            f"{chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Übrige leere Subplots ausblenden
    y_max = max(ax.get_ylim()[1] for ax in axes_flat1[:n_rivers])
    y_min = min(ax.get_ylim()[0] for ax in axes_flat1[:n_rivers])

    for ax in axes_flat1[:n_rivers]:
        ax.set_ylim(y_min, y_max, auto=False)

    for ax in axes_flat1[n_rivers:]:
        ax.set_visible(False)

    # fig.suptitle("Jährliche Anomalie der Abflussspende", fontsize=11)
    plt.tight_layout()
    plt.savefig("./results/river_basin_annual_anomalies.png", dpi=300)
    plt.close(fig)

    fig4, axes4 = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6, nrows * 5),  # Höhe pro Zeile erhöht
        constrained_layout=True,  # statt tight_layout
    )
    axes_flat4 = axes4.flatten()

    for ax, (river, (scenario, control)) in zip(
        axes_flat4, annual_snowmelt_anomaly_data.items()
    ):
        plot_annual_anomaly(scenario, control, ax=ax, title="Snowmelt Anomaly")

    # Übrige leere Subplots ausblenden
    y_max = max(ax.get_ylim()[1] for ax in axes_flat4[:n_rivers])
    y_min = min(ax.get_ylim()[0] for ax in axes_flat4[:n_rivers])

    for ax in axes_flat4[:n_rivers]:
        ax.set_ylim(y_min, y_max, auto=False)

    for ax in axes_flat4[n_rivers:]:
        ax.set_visible(False)

    # fig.suptitle("Jährliche Anomalie der Abflussspende", fontsize=11)
    plt.tight_layout()
    plt.savefig("./results/river_basin_annual_snowmelt_anomalies.png", dpi=300)
    plt.close(fig4)

    fig2, axes2 = plt.subplots(
        nrows,
        ncols,
        figsize=(ncols * 6, nrows * 5),
        constrained_layout=True,
    )
    axes_flat2 = axes2.flatten()

    for ax, (river, (scenario, control)) in zip(
        axes_flat2, annual_total_discharge_anomaly_data.items()
    ):
        plot_annual_sums_two_bars(scenario, control, ax=ax)

    for ax in axes_flat2[n_rivers:]:
        ax.set_visible(False)

    plt.savefig("./results/river_basin_annual_sums.png", dpi=300)
    plt.close(fig2)

    fig3, axes3 = plt.subplots(
        3,
        2,
        figsize=(2 * 6, 3 * 5),
        constrained_layout=True,
    )
    axes_flat3 = axes3.flatten()  # ← eigener Name

    for ax, (river, river_data), idx in zip(
        axes_flat3, monthly_snow_rain_data.items(), range(len(monthly_snow_rain_data))
    ):
        plot_monthly_bars_rain_snow(
            river_data["scenario_snowmelt"],
            river_data["scenario_rain"],
            river_data["control_snowmelt"],
            river_data["control_rain"],
            river,
            ax=ax,
        )
        ax.text(
            -0.15,
            1.05,
            f"{chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Nach allen Plots die Y-Achse pro Subplot symmetrieren
    # Nach allen Plots
    for ax in axes_flat3[:n_rivers]:
        y_min, y_max = ax.get_ylim()
        y_abs_max = max(abs(y_min), abs(y_max))
        ax.set_ylim(-y_abs_max, y_abs_max)
        # Y-Tick-Labels neu setzen nach set_ylim
        yticks = ax.get_yticks()
        ax.set_yticklabels([str(abs(int(y))) for y in yticks])

    plt.savefig("./results/basin/all_rivers_monthly_bars_rain_snow.png", dpi=300)
    plt.close(fig3)
