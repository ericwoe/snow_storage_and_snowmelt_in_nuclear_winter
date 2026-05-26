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