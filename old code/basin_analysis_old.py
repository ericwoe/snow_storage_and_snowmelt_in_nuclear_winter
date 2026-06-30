def plot_monthly_bars(scenario, control, river, save_dir="./results/basin"):
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
    plt.savefig(os.path.join(save_dir, f"{river}_monthly_bars.png"), dpi=300)
    plt.close(fig)

def plot_monthly_anomaly(scenario, control, river):
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

#Annual 
fig2, axes2 = plt.subplots(
    nrows, ncols, figsize=(ncols * 6, nrows * 5), constrained_layout=True
)
axes_flat2 = axes2.flatten()

for ax, (river, (scenario, control)) in zip(
    axes_flat2, annual_total_discharge_anomaly_data.items()
):
    plot_annual_sums_two_bars(scenario, control, river=river, ax=ax)

for ax in axes_flat2[n_rivers:]:
    ax.set_visible(False)

plt.savefig(basin_analysis_dir / "river_basin_annual_sums.png", dpi=300)
plt.close(fig2)

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

def weighted_monthly_mean(da, mask):
    return (da * mask).sum(dim=["lat", "lon"]) / mask.sum(dim=["lat", "lon"])

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

  "Donau": [
            "./data/HydroBasins/hybas_eu_lev01-12_v1c/hybas_eu_lev04_v1c.shp",
            2040008490,
        ],

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

    cell_area = compute_grid_cell_area(ds_control.snow_storage.isel(time=0))

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
        rain_47 = ds_47.rain.where(river_mask > 0)
        rain_control = ds_control.rain.where(river_mask > 0)

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
            river=river,
            title="Snowmelt",
        )
        plot_annual_anomaly(
            dsc_47_rain_annual_sum, dsc_control_rain_annual_sum, river=river, title="Rain"
        )
        plot_annual_anomaly(
            dsc_47_snowmelt_annual_sum + dsc_47_rain_annual_sum,
            dsc_control_snowmelt_annual_sum + dsc_control_rain_annual_sum,
            river=river,
            title="Total Discharge",
        )

        plot_annual_anomaly(
            dsc_47_precip_annual_sum,
            dsc_control_precip_annual_sum,
            river=river,
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
        plot_monthly_bars(dsc_47_m3_s, dsc_control_m3_s, river=river)

    print(annual_snowmelt_anomaly_data)
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
        plot_annual_anomaly(scenario, control, river=river, ax=ax, title="Annual Total Discharge")
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
        figsize=(ncols * 6, nrows * 5),
        constrained_layout=True,
    )
    axes_flat4 = axes4.flatten()

    for ax, (river, (scenario, control)), idx in zip(
        axes_flat4,
        annual_snowmelt_anomaly_data.items(),
        range(len(annual_snowmelt_anomaly_data)),
    ):
        plot_annual_anomaly(scenario, control, river=river, ax=ax, title="Snowmelt")
        ax.text(
            -0.09,
            1.04,
            f"{chr(97 + idx)})",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            weight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # Übrige leere Subplots ausblenden
    for ax in axes_flat4[n_rivers:]:
        ax.set_visible(False)

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
        plot_annual_sums_two_bars(scenario, control, river=river, ax=ax)

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
