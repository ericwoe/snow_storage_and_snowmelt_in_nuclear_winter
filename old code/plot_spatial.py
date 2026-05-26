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
