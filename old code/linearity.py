import xarray as xr
import numpy as np
from postprocessing.snow_analysis import (
    compute_cell_area,
    sum_per_month,
    snow_covered_area_proportion_monthly,
)
from matplotlib import pyplot as plt

max_radiation_anomaly = {16: -31.1, 47: -68.7, 150: -115.3}  # aus Harrison Preprint


def plot_linearity_analysis_mean(*datasets, control=None, cell_area=None, mask=None):
    """
    Prüft, ob die Schneereaktion linear mit der Rußmenge skaliert.
    """
    from scipy.stats import linregress

    forcing = []
    response_storage = []
    response_cover = []
    labels = []

    # Control als Basislinie
    ctrl_storage = sum_per_month(control.snow_storage, cell_area, mask).mean().values
    ctrl_cover = (
        snow_covered_area_proportion_monthly(control.snow_storage, cell_area, mask)
        .mean()
        .values
    )

    for ds in datasets:
        tg = ds.attrs["soot"]
        forcing.append(max_radiation_anomaly[tg])
        labels.append(f"{tg} Tg")

        stor = sum_per_month(ds.snow_storage, cell_area, mask).mean().values
        cov = (
            snow_covered_area_proportion_monthly(ds.snow_storage, cell_area, mask)
            .mean()
            .values
        )

        response_storage.append(stor - ctrl_storage)
        response_cover.append(cov - ctrl_cover)

    forcing = np.array(forcing)
    response_storage = np.array(response_storage)
    response_cover = np.array(response_cover)

    # Lineare Regression für beide
    slope_s, intercept_s, r_value_s, _, _ = linregress(forcing, response_storage)
    slope_c, intercept_c, r_value_c, _, _ = linregress(forcing, response_cover)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Snow Storage Anomaly ---
    ax = axes[0]
    ax.scatter(forcing, response_storage, s=100, zorder=5)
    for f, r, l in zip(forcing, response_storage, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    x_fit = np.linspace(min(forcing.min(), 0) * 1.1, max(forcing.max(), 0) * 1.1, 100)
    ax.plot(
        x_fit,
        slope_s * x_fit + intercept_s,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_value_s**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow storage [m\u00b3]")
    ax.set_title("Snow Storage Response")
    ax.legend()

    # --- Plot 2: Snow Cover Anomaly ---
    ax = axes[1]
    ax.scatter(forcing, response_cover, s=100, zorder=5)
    for f, r, l in zip(forcing, response_cover, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    x_fit = np.linspace(min(forcing.min(), 0) * 1.1, max(forcing.max(), 0) * 1.1, 100)
    ax.plot(
        x_fit,
        slope_c * x_fit + intercept_c,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_value_c**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow covered area [pp]")
    ax.set_title("Snow Cover Response")
    ax.legend()

    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/linearity_analysis_mean_anomaly.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_linearity_analysis_max(*datasets, control=None, cell_area=None, mask=None):
    """
    Prüft, ob die maximale Schneereaktion linear mit der Rußmenge skaliert.

    Statt des zeitlichen Mittelwerts wird das Maximum der Anomalie verwendet,
    da manche Zellen im Modell dauerhaft Schnee akkumulieren und so den
    Mittelwert verzerren. Das Maximum zeigt den stärksten Effekt des
    nuklearen Winters auf die Schneebedeckung.

    Erstellt zwei Scatter-Plots:
        - Links: Max. Snow Storage Anomalie vs. Rußmenge
        - Rechts: Max. Snow Cover Anomalie vs. Rußmenge
    """
    from scipy.stats import linregress

    forcing = []
    response_storage = []
    response_cover = []
    labels = []

    # ---------------------------------------------------------------
    # Schritt 1: Zeitreihen für Control berechnen
    # ---------------------------------------------------------------

    # Globales Schneevolumen pro Monat im Control (Zeitreihe)
    ctrl_storage_ts = sum_per_month(control.snow_storage, cell_area, mask)

    # Globale Schneebedeckung pro Monat im Control (Zeitreihe)
    ctrl_cover_ts = snow_covered_area_proportion_monthly(
        control.snow_storage, cell_area, mask
    )

    # ---------------------------------------------------------------
    # Schritt 2: Für jedes Szenario die maximale Anomalie berechnen
    # ---------------------------------------------------------------

    for ds in datasets:
        tg = ds.attrs["soot"]
        forcing.append(max_radiation_anomaly[tg])
        labels.append(f"{tg} Tg")

        # Zeitreihe für das Szenario
        stor_ts = sum_per_month(ds.snow_storage, cell_area, mask)
        cov_ts = snow_covered_area_proportion_monthly(ds.snow_storage, cell_area, mask)

        # Anomalie-Zeitreihe: Szenario minus Control (pro Monat)
        storage_anomaly_ts = stor_ts - ctrl_storage_ts
        cover_anomaly_ts = cov_ts - ctrl_cover_ts

        # Maximum der Anomalie über die gesamte Zeitreihe
        # = der Monat mit dem stärksten Schnee-Zuwachs gegenüber Control
        response_storage.append(storage_anomaly_ts.max().values)
        response_cover.append(cover_anomaly_ts.max().values)

    forcing = np.array(forcing)
    response_storage = np.array(response_storage)
    response_cover = np.array(response_cover)

    # ---------------------------------------------------------------
    # Schritt 3: Lineare Regression
    # ---------------------------------------------------------------

    slope_s, intercept_s, r_value_s, _, _ = linregress(forcing, response_storage)
    slope_c, intercept_c, r_value_c, _, _ = linregress(forcing, response_cover)

    # ---------------------------------------------------------------
    # Schritt 4: Plotten
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Max Snow Storage Anomaly ---
    ax = axes[0]
    ax.scatter(forcing, response_storage, s=100, zorder=5)
    for f, r, l in zip(forcing, response_storage, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    x_fit = np.linspace(min(forcing.min(), 0) * 1.1, max(forcing.max(), 0) * 1.1, 100)
    ax.plot(
        x_fit,
        slope_s * x_fit + intercept_s,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_value_s**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow storage (max) [m\u00b3]")
    ax.set_title("Max. Snow Storage Response")
    ax.legend()

    # --- Plot 2: Max Snow Cover Anomaly ---
    ax = axes[1]
    ax.scatter(forcing, response_cover, s=100, zorder=5)
    for f, r, l in zip(forcing, response_cover, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    x_fit = np.linspace(min(forcing.min(), 0) * 1.1, max(forcing.max(), 0) * 1.1, 100)
    ax.plot(
        x_fit,
        slope_c * x_fit + intercept_c,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_value_c**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow covered area (max) [pp]")
    ax.set_title("Max. Snow Cover Response")
    ax.legend()

    # ---------------------------------------------------------------
    # Schritt 5: Speichern
    # ---------------------------------------------------------------

    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/linearity_analysis_max_anomaly.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_linearity_analysis_min_max_cover(
    *datasets, control=None, cell_area=None, mask=None
):
    """
    Prüft, ob die Schneereaktion linear mit der Rußmenge skaliert.

    Verwendet Maximum und Minimum der Anomalie statt des Mittelwerts,
    da manche Zellen im Modell dauerhaft Schnee akkumulieren.

    Erstellt drei Scatter-Plots:
      - Links: Max. Snow Storage Anomalie vs. Rußmenge (stärkster Zuwachs)
      - Mitte: Min. Snow Storage Anomalie vs. Rußmenge (stärkster Verlust)
      - Rechts: Max. Snow Cover Anomalie vs. Rußmenge
    """
    from scipy.stats import linregress

    forcing = []
    response_storage_max = []
    response_storage_min = []
    response_cover_max = []
    labels = []

    # ---------------------------------------------------------------
    # Schritt 1: Zeitreihen für Control berechnen
    # ---------------------------------------------------------------

    ctrl_storage_ts = sum_per_month(control.snow_storage, cell_area, mask)
    ctrl_cover_ts = snow_covered_area_proportion_monthly(
        control.snow_storage, cell_area, mask
    )

    # ---------------------------------------------------------------
    # Schritt 2: Für jedes Szenario Max und Min der Anomalie berechnen
    # ---------------------------------------------------------------

    for ds in datasets:
        tg = ds.attrs["soot"]
        forcing.append(max_radiation_anomaly[tg])
        labels.append(f"{tg} Tg")

        # Zeitreihen für das Szenario
        stor_ts = sum_per_month(ds.snow_storage, cell_area, mask)
        cov_ts = snow_covered_area_proportion_monthly(ds.snow_storage, cell_area, mask)

        # Anomalie-Zeitreihe: Szenario minus Control
        storage_anomaly_ts = stor_ts - ctrl_storage_ts
        cover_anomaly_ts = cov_ts - ctrl_cover_ts

        # Maximum = Monat mit stärkstem Schnee-Zuwachs
        response_storage_max.append(storage_anomaly_ts.max().values)

        # Minimum = Monat mit stärkstem Schnee-Verlust
        # (z.B. durch veränderte Niederschlagsmuster oder Schmelze)
        response_storage_min.append(storage_anomaly_ts.min().values)

        # Maximum der Schneebedeckungs-Anomalie
        response_cover_max.append(cover_anomaly_ts.max().values)

    forcing = np.array(forcing)
    response_storage_max = np.array(response_storage_max)
    response_storage_min = np.array(response_storage_min)
    response_cover_max = np.array(response_cover_max)

    # ---------------------------------------------------------------
    # Schritt 3: Lineare Regression für alle drei
    # ---------------------------------------------------------------

    slope_max, intercept_max, r_max, _, _ = linregress(forcing, response_storage_max)
    slope_min, intercept_min, r_min, _, _ = linregress(forcing, response_storage_min)
    slope_cov, intercept_cov, r_cov, _, _ = linregress(forcing, response_cover_max)

    # ---------------------------------------------------------------
    # Schritt 4: Plotten
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Plot 1: Max Snow Storage Anomaly (Zuwachs) ---
    ax = axes[0]
    ax.scatter(forcing, response_storage_max, s=100, zorder=5, color="tab:blue")
    for f, r, l in zip(forcing, response_storage_max, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    x_fit = np.linspace(min(forcing.min(), 0) * 1.1, max(forcing.max(), 0) * 1.1, 100)
    ax.plot(
        x_fit,
        slope_max * x_fit + intercept_max,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_max**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow storage (max) [m\u00b3]")
    ax.set_title("Peak Snow Storage Increase")
    ax.legend()

    # --- Plot 2: Min Snow Storage Anomaly (Verlust) ---
    ax = axes[1]
    ax.scatter(forcing, response_storage_min, s=100, zorder=5, color="tab:red")
    for f, r, l in zip(forcing, response_storage_min, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    ax.plot(
        x_fit,
        slope_min * x_fit + intercept_min,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_min**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow storage (min) [m\u00b3]")
    ax.set_title("Peak Snow Storage Decrease")
    ax.legend()

    # --- Plot 3: Max Snow Cover Anomaly ---
    ax = axes[2]
    ax.scatter(forcing, response_cover_max, s=100, zorder=5, color="tab:green")
    for f, r, l in zip(forcing, response_cover_max, labels):
        ax.annotate(l, (f, r), textcoords="offset points", xytext=(10, 5))

    ax.plot(
        x_fit,
        slope_cov * x_fit + intercept_cov,
        "--",
        color="gray",
        label=f"Linear fit (R²={r_cov**2:.3f})",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
    ax.set_ylabel("\u0394 Snow covered area (max) [pp]")
    ax.set_title("Peak Snow Cover Increase")
    ax.legend()

    # ---------------------------------------------------------------
    # Schritt 5: Speichern
    # ---------------------------------------------------------------

    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/linearity_analysis_min_max_mean.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_linearity_analysis_absolut(*datasets, control=None, cell_area=None, mask=None):
    """
    Prüft, ob die Schneereaktion linear mit der Rußmenge skaliert.

    Verwendet absolute Werte (kein Anomalie-Vergleich zu Control).
    Control wird als Punkt bei 0 Tg eingezeichnet.

    Erstellt vier Scatter-Plots:
      - Oben links:  Max. Snow Storage vs. Rußmenge
      - Oben rechts: Mean Snow Storage vs. Rußmenge
      - Unten links:  Max. Snow Cover vs. Rußmenge
      - Unten rechts: Mean Snow Cover vs. Rußmenge
    """
    from scipy.stats import linregress

    forcing = [0]
    response_storage_max = []
    response_storage_mean = []
    response_cover_max = []
    response_cover_mean = []
    labels = ["Control"]

    # ---------------------------------------------------------------
    # Schritt 1: Control berechnen
    # ---------------------------------------------------------------

    ctrl_storage_ts = sum_per_month(control.snow_storage, cell_area, mask)
    ctrl_cover_ts = snow_covered_area_proportion_monthly(
        control.snow_storage, cell_area, mask
    )

    response_storage_max.append(ctrl_storage_ts.max().values)
    response_storage_mean.append(ctrl_storage_ts.mean().values)
    response_cover_max.append(ctrl_cover_ts.max().values)
    response_cover_mean.append(ctrl_cover_ts.mean().values)

    # ---------------------------------------------------------------
    # Schritt 2: Für jedes Szenario Metriken berechnen
    # ---------------------------------------------------------------

    for ds in datasets:
        tg = ds.attrs["soot"]
        forcing.append(max_radiation_anomaly[tg])
        labels.append(f"{tg} Tg")

        stor_ts = sum_per_month(ds.snow_storage, cell_area, mask)
        cov_ts = snow_covered_area_proportion_monthly(ds.snow_storage, cell_area, mask)

        response_storage_max.append(stor_ts.max().values)
        response_storage_mean.append(stor_ts.mean().values)
        response_cover_max.append(cov_ts.max().values)
        response_cover_mean.append(cov_ts.mean().values)

    forcing = np.array(forcing)
    response_storage_max = np.array(response_storage_max)
    response_storage_mean = np.array(response_storage_mean)
    response_cover_max = np.array(response_cover_max)
    response_cover_mean = np.array(response_cover_mean)

    # ---------------------------------------------------------------
    # Schritt 3: Lineare Regression für alle vier
    # ---------------------------------------------------------------

    def fit(x, y):
        slope, intercept, r, _, _ = linregress(x, y)
        return slope, intercept, r

    slope_smax, intercept_smax, r_smax = fit(forcing, response_storage_max)
    slope_smean, intercept_smean, r_smean = fit(forcing, response_storage_mean)
    slope_cmax, intercept_cmax, r_cmax = fit(forcing, response_cover_max)
    slope_cmean, intercept_cmean, r_cmean = fit(forcing, response_cover_mean)

    # ---------------------------------------------------------------
    # Schritt 4: Plotten
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    x_fit = np.linspace(min(forcing.min(), 0) * 1.1, max(forcing.max(), 0) * 1.1, 100)

    plot_configs = [
        (
            axes[0, 0],
            response_storage_max,
            slope_smax,
            intercept_smax,
            r_smax,
            "tab:blue",
            "Peak Snow Storage",
            "Snow storage (max) [m³]",
        ),
        (
            axes[0, 1],
            response_storage_mean,
            slope_smean,
            intercept_smean,
            r_smean,
            "tab:orange",
            "Mean Snow Storage",
            "Snow storage (mean) [m³]",
        ),
        (
            axes[1, 0],
            response_cover_max,
            slope_cmax,
            intercept_cmax,
            r_cmax,
            "tab:green",
            "Peak Snow Cover",
            "Snow covered area (max) [pp]",
        ),
        (
            axes[1, 1],
            response_cover_mean,
            slope_cmean,
            intercept_cmean,
            r_cmean,
            "tab:red",
            "Mean Snow Cover",
            "Snow covered area (mean) [pp]",
        ),
    ]

    for ax, response, slope, intercept, r, color, title, ylabel in plot_configs:
        ax.scatter(forcing, response, s=100, zorder=5, color=color)
        for f, r_val, l in zip(forcing, response, labels):
            ax.annotate(l, (f, r_val), textcoords="offset points", xytext=(10, 5))
        ax.plot(
            x_fit,
            slope * x_fit + intercept,
            "--",
            color="gray",
            label=f"Linear fit (R²={r**2:.3f})",
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Radiative Forcing [W/m\u00b2]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()

    # ---------------------------------------------------------------
    # Schritt 5: Speichern
    # ---------------------------------------------------------------

    plt.tight_layout()
    fig.savefig(
        "./results/intercomparison/linearity_analysis_absolut.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


if __name__ == "__main__":

    ds_16 = xr.open_dataset("./results/16/snow_16.nc")
    ds_16.attrs["soot"] = 16
    ds_47 = xr.open_dataset("./results/47/snow_47.nc")
    ds_47.attrs["soot"] = 47
    ds_150 = xr.open_dataset("./results/150/snow_150.nc")
    ds_150.attrs["soot"] = 150
    ds_ctrl = xr.open_dataset("./results/Control/snow_control.nc")
    ds_ctrl.attrs["soot"] = 0
    mask = xr.open_dataarray("./data/interim/land_mask_neu.nc")

    cell_area = compute_cell_area(ds_ctrl.snow_storage)

    plot_linearity_analysis_mean(
        ds_16, ds_47, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )

    plot_linearity_analysis_max(
        ds_16, ds_47, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )
    plot_linearity_analysis_min_max_cover(
        ds_16, ds_47, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )

    plot_linearity_analysis_absolut(
        ds_16, ds_47, ds_150, control=ds_ctrl, cell_area=cell_area, mask=mask
    )
