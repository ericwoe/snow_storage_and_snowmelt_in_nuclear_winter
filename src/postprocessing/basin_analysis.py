import sys
import os
import numpy as np
import cftime
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


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
    scenario, control, river, labels=["Scenario", "Control"], ax=None
):
    n_years = len(scenario)
    year_indices = np.arange(n_years)

    bar_width = 0.4
    offset = bar_width / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        standalone = True
    else:
        standalone = False

    control_mean = control.mean(dim="year")

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


def plot_annual_anomaly(
    annual_scenario_ts,
    annual_control_ts,
    river,
    ax=None,
    title=None,
    save_dir="./results/basin",
):
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
        plt.savefig(
            os.path.join(save_dir, f"{river}_annual_{title}_sum_anomaly.png"), dpi=600
        )
        plt.close(fig)


def plot_monthly_bars_rain_snow(
    scenario_snowmelt,
    scenario_rain,
    control_snowmelt,
    control_rain,
    river,
    ax=None,
    save_dir="./results/basin",
):
    import matplotlib.pyplot as plt
    import numpy as np

    standalone = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
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
    ax.set_yticks(yticks)
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
        plt.savefig(
            os.path.join(save_dir, f"{river}_monthly_bars_rain_snow.png"), dpi=350
        )
        plt.close()
