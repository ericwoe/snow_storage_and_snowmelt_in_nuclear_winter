import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import numpy as np
import cftime
import xarray as xr


def plot_snow_anomaly_at_peak_times_old(
    scenarios: dict,
    control,
    peak_times: dict,
    variable: str = "snow_storage",
):
    """
    Für jede Metrik (anomaly_sum, abs_anomaly_sum, positive_anomaly_sum) wird
    eine Figure mit 3 Subplots (je ein Szenario) erzeugt. Jeder Subplot zeigt
    die räumliche Anomalie des Szenarios zum Control zum jeweiligen Peak-Zeitpunkt.

    Parameters
    ----------
    scenarios       : dict  {name: dataset}
    control         : xr.Dataset
    peak_times      : dict  {metric: {scenario_name: cftime}}
    variable        : str   Variable im Dataset
    """
    colors = [
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
        "#02203a",
    ]
    cmap = mcolors.ListedColormap(colors)
    bin_edges = np.array([-5, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 5])
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=len(bin_edges) - 1)

    metric_labels = {
        "anomaly_sum": "Anomalie",
        "abs_anomaly_sum": "Summe absoluter Anomalien",
        "positive_anomaly_sum": "Summe positiver Anomalien",
    }

    for metric, times_per_scenario in peak_times.items():

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(24, 7),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        fig.suptitle(
            f"Snow Storage Anomaly at Peak Time – {metric_labels[metric]}",
            fontsize=16,
        )

        for ax, (scen_name, ds) in zip(axes, scenarios.items()):
            t = times_per_scenario[scen_name]

            # Anomalie zum Control am selben Zeitpunkt
            anomaly = (
                ds[variable].sel(time=t) - control[variable].sel(time=t)
            ) / 1000  # mm → m

            # Nullwerte maskieren
            anomaly_masked = anomaly.where(anomaly != 0)

            ax.add_feature(cfeature.OCEAN, color="white")
            im = anomaly_masked.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
            )
            ax.coastlines(linewidth=0.5, zorder=11)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
            ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
            ax.set_title(
                f"{scen_name}  –  Jahr {t.year}, Monat {t.month}",
                fontsize=12,
            )

            cbar = plt.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.02,
                pad=0.02,
                extend="both",
            )
            cbar.set_label("Snow storage anomaly [m]", fontsize=11)
            cbar.set_ticks(bin_edges)

        plt.tight_layout()
        fig.savefig(
            f"./results/intercomparison/snow_anomaly_at_peak_{metric}.png",
            dpi=150,
            bbox_inches="tight",
        )

        print(f"Gespeichert: snow_anomaly_at_peak_{metric}.png")


def find_abs_anomaly_peak_time(
    scenario: xr.Dataset,
    control: xr.Dataset,
    variable: str = "snow_storage",
):
    """
    Identify the time point of peak absolute anomaly magnitude at any
    individual grid cell, relative to the control run's monthly
    climatology, for a single scenario.

    The anomaly is computed grid cell-wise, relative to the control run's
    monthly climatology mean at each grid cell (i.e., the control's mean
    for the corresponding calendar month at that location). This removes
    normal seasonal and interannual variability in the control run before
    identifying the peak deviation. The returned time point is the one at
    which the largest such anomaly occurs anywhere in the spatial domain.

    Parameters
    ----------
    scenario : xr.Dataset
        Scenario dataset containing `variable` with dimensions
        ("time", "lat", "lon").
    control : xr.Dataset
        Control run dataset, structured like `scenario` and sharing the
        same time coordinate values.
    variable : str, optional
        Name of the variable in `scenario` and `control` to analyze.
        Defaults to "snow_storage".

    Returns
    -------
    time_abs_peak : cftime
        Time value at which the largest grid cell-wise climatological
        anomaly magnitude, across all grid cells, is reached.

    Notes
    -----
    `scenario` and `control` should already be restricted to the spatial
    domain of interest (e.g., a specific cluster), since this function
    does not perform any cluster filtering itself. Unlike
    `find_mean_anomaly_peak_time`, this function does not require
    `cell_area` or `mask`, since no spatial averaging is performed here.
    """
    control_monthly_grid = control[variable].groupby("time.month").mean()

    anomaly_grid = (
        scenario[variable].groupby("time.month") - control_monthly_grid
    ).sortby("time")
    abs_anomaly_grid = np.abs(anomaly_grid)

    max_abs_anomaly_per_time = abs_anomaly_grid.max(dim=["lat", "lon"])
    time_abs_peak = max_abs_anomaly_per_time.idxmax(dim="time").item()

    return time_abs_peak


def plot_monthly_snow_anomaly(
    scenario: xr.Dataset,
    control: xr.Dataset,
    scenario_name: str,
    variable: str = "snow_storage",
    output_dir: str = "./results/intercomparison/spatial_anomalies",
    vmin: float = -5,
    vmax: float = 5,
):
    """
    Plot the spatial snow storage anomaly between a scenario and the control
    run for every available month, saving one figure per month.

    For each time step present in `scenario`, this function computes the
    grid cell-wise difference between the scenario and the control run
    (scenario minus control) and plots it on a Robinson projection map using
    a continuous diverging color scale centered at zero. Each monthly plot
    is saved as a separate PNG file in `output_dir`, named using the
    scenario name and the simulation year/month.

    Parameters
    ----------
    scenario : xr.Dataset
        Scenario dataset containing `variable` with dimensions
        ("time", "lat", "lon").
    control : xr.Dataset
        Control run dataset, structured like `scenario` and sharing the
        same time coordinate values.
    scenario_name : str
        Name of the scenario, used in the plot title and output filename
        (e.g., "47 Tg").
    variable : str, optional
        Name of the variable in `scenario` and `control` to analyze.
        Defaults to "snow_storage".
    output_dir : str, optional
        Directory where the monthly figures are saved. Created if it does
        not already exist.
    vmin, vmax : float, optional
        Lower and upper bounds of the color scale, in meters. The scale is
        symmetric around zero; values are clipped visually at these bounds
        via colorbar extension arrows. Defaults to -5 and 5.

    Notes
    -----
    The anomaly is computed in millimeters from the input data and
    converted to meters for plotting. Grid cells with zero anomaly are
    masked (not colored) to visually emphasize regions with an actual
    deviation from the control. The color scale uses `TwoSlopeNorm` to
    ensure that zero is always mapped to the center of the diverging
    colormap, even though `vmin` and `vmax` here are symmetric.
    """
    os.makedirs(output_dir, exist_ok=True)

    cmap = plt.get_cmap("RdBu")
    norm = SymLogNorm(linthresh=0.5, vmin=vmin, vmax=vmax, base=10)

    for t in scenario.time.values[12::2]:
        # Anomaly relative to the control run at the same time step,
        # converted from mm to m
        anomaly = (
            scenario[variable].sel(time=t) - control[variable].sel(time=t)
        ) / 1000

        # Mask zero-valued cells so they are not colored, emphasizing
        # regions with an actual deviation from the control
        anomaly_masked = anomaly.where(anomaly != 0)

        fig, ax = plt.subplots(
            figsize=(10, 6),
            subplot_kw={"projection": ccrs.Robinson()},
        )

        ax.add_feature(cfeature.OCEAN, color="white")
        im = anomaly_masked.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            norm=norm,
            add_colorbar=False,
        )
        ax.coastlines(linewidth=0.5, zorder=11)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
        ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
        ax.set_title(
            f"{scenario_name} – Year {t.year}, Month {t.month}",
            fontsize=12,
        )

        cbar = plt.colorbar(
            im,
            ax=ax,
            orientation="vertical",
            fraction=0.02,
            pad=0.02,
            extend="both",
        )
        cbar.set_label("Snow storage anomaly [m]", fontsize=11)

        # Explicit, symmetric ticks that make the nonlinear (SymLog) scale
        # easier to read: a fine-grained linear region near zero, transitioning
        # to coarser logarithmic steps for larger anomalies
        tick_values = [-5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5]
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([f"{v:g}" for v in tick_values])

        plt.tight_layout()

        # Filename includes scenario name, year, and zero-padded month for
        # correct chronological sorting (e.g., "_03_" rather than "_3_")
        filename = (
            f"snow_anomaly_{scenario_name}_year{t.year:02d}_month{t.month:02d}.png"
        )
        fig.savefig(
            os.path.join(output_dir, filename),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_snow_anomaly_at_peak_times(
    scenarios: dict,
    control,
    peak_times: dict,
    variable: str = "snow_storage",
):
    """
    Für jedes Szenario wird eine Figure mit 2×2 Subplots erzeugt.
    Die ersten 3 Subplots zeigen die räumliche Anomalie zu den jeweiligen
    Peak-Zeitpunkten der 3 Metriken, der vierte Subplot bleibt leer.

    Parameters
    ----------
    scenarios       : dict  {name: dataset}
    control         : xr.Dataset
    peak_times      : dict  {metric: {scenario_name: cftime}}
    variable        : str   Variable im Dataset
    """
    colors = [
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
        "#02203a",
    ]
    cmap = mcolors.ListedColormap(colors)
    bin_edges = np.array([-5, -4, -2, -1, -0.1, 0, 0.1, 1, 2, 4, 5])
    norm = mcolors.BoundaryNorm(bin_edges, ncolors=len(bin_edges) - 1)

    metric_labels = {
        "anomaly_sum": "Anomalie",
        "abs_anomaly_sum": "Summe absoluter Anomalien",
        "positive_anomaly_sum": "Summe positiver Anomalien",
    }

    # Äußere Schleife: je Figure pro Szenario
    for scen_name, ds in scenarios.items():

        fig, axes = plt.subplots(
            2,
            2,
            figsize=(22, 12),
            subplot_kw={"projection": ccrs.Robinson()},
        )
        fig.suptitle(
            f"Snow Storage Anomaly at Peak Time – {scen_name}",
            fontsize=16,
        )

        # Innere Schleife: je Subplot pro Metrik
        for ax, (metric, times_per_scenario) in zip(axes.flat, peak_times.items()):
            t = times_per_scenario[scen_name]

            # Anomalie zum Control am selben Zeitpunkt
            anomaly = (
                ds[variable].sel(time=t) - control[variable].sel(time=t)
            ) / 1000  # mm → m

            # Nullwerte maskieren
            anomaly_masked = anomaly.where(anomaly != 0)

            ax.add_feature(cfeature.OCEAN, color="white")
            im = anomaly_masked.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=cmap,
                norm=norm,
                add_colorbar=False,
            )
            ax.coastlines(linewidth=0.5, zorder=11)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, zorder=11)
            ax.add_feature(cfeature.OCEAN, color="lightgrey", zorder=10)
            ax.set_title(
                f"{metric_labels[metric]}  –  Jahr {t.year}, Monat {t.month}",
                fontsize=12,
            )

            cbar = plt.colorbar(
                im,
                ax=ax,
                orientation="vertical",
                fraction=0.02,
                pad=0.02,
                extend="both",
            )
            cbar.set_label("Snow storage anomaly [m]", fontsize=11)
            cbar.set_ticks(bin_edges)

        # Vierten Subplot ausblenden
        axes[1, 1].set_visible(False)

        plt.tight_layout()
        fig.savefig(
            f"./results/intercomparison/snow_anomaly_at_peak_{scen_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        print(f"Gespeichert: snow_anomaly_at_peak_{scen_name}.png")


# ── Aufruf ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plt.rcParams.update(
        {
            "axes.edgecolor": "black",
            "axes.linewidth": 0.2,
        }
    )
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_ctrl = xr.open_dataset("./results/Control/snow_control.nc")

    # ── Zeitpunkte der Maxima pro Metrik und Szenario ────────────────────────
    PEAK_TIMES = {
        "anomaly_sum": {
            "16_Tg": cftime.DatetimeNoLeap(7, 5, 1, has_year_zero=True),
            "47_Tg": cftime.DatetimeNoLeap(7, 5, 1, has_year_zero=True),
            "150_Tg": cftime.DatetimeNoLeap(8, 6, 1, has_year_zero=True),
        },
        "abs_anomaly_sum": {
            "16_Tg": cftime.DatetimeNoLeap(18, 3, 1, has_year_zero=True),
            "47_Tg": cftime.DatetimeNoLeap(12, 3, 1, has_year_zero=True),
            "150_Tg": cftime.DatetimeNoLeap(12, 3, 1, has_year_zero=True),
        },
        "positive_anomaly_sum": {
            "16_Tg": cftime.DatetimeNoLeap(10, 3, 1, has_year_zero=True),
            "47_Tg": cftime.DatetimeNoLeap(12, 5, 1, has_year_zero=True),
            "150_Tg": cftime.DatetimeNoLeap(8, 6, 1, has_year_zero=True),
        },
    }

    # ── Szenarien ─────────────────────────────────────────────────────────────
    SCENARIOS = {
        "16_Tg": ds_16,
        "47_Tg": ds_47,
        "150_Tg": ds_150,
    }

    plot_snow_anomaly_at_peak_times(
        scenarios=SCENARIOS,
        control=ds_ctrl,
        peak_times=PEAK_TIMES,
        variable="snow_storage",
    )

    plot_snow_anomaly_at_peak_times_old(
        scenarios=SCENARIOS,
        control=ds_ctrl,
        peak_times=PEAK_TIMES,
        variable="snow_storage",
    )
