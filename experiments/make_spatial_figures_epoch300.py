"""Create source-traceable spatial figures for the ASDKL manuscript.

Figure contract
---------------
Core conclusions:
1. The synthetic generator contains localized structure at multiple spatial scales.
2. On the same held-out LUCAS split, ASDKL follows the observed spatial pattern at
   least as closely as HistGBR, the second-ranked full-data comparator.
3. ASDKL also reconstructs the main full-data California pattern, while the
   residual map makes the remaining localized errors visible.
4. No component removal improves mean RMSE on both moderate-size datasets;
   the effect size remains dataset dependent.

All plotted prediction panels use held-out test observations only.  Common colour
limits are used within each truth/prediction comparison.  Display clipping is
limited to the 1st and 99th percentiles and does not affect reported metrics.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


WORK = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data"
RESULTS = PROJECT / "results" / "revision14_shared_relation" / "six_block_reference"
FIGURE_OUT = PROJECT / "results" / "generated_figures"
SOURCE_OUT = PROJECT / "results" / "spatial_figure_source_data"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)
SOURCE_OUT.mkdir(parents=True, exist_ok=True)

BLUE = "#2C6F93"
ORANGE = "#D97732"
RED = "#B6483D"
GREY = "#9AA7B1"
GRID = "#E5EAEE"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def export(fig: mpl.figure.Figure, stem: str) -> None:
    for extension in ("svg", "pdf", "png", "tiff"):
        dpi = 600 if extension == "tiff" else 300
        fig.savefig(
            FIGURE_OUT / f"{stem}.{extension}",
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    text_method = ax.text2D if hasattr(ax, "text2D") else ax.text
    text_method(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.6),
    )


def local_spectral_field(coordinates: np.ndarray) -> np.ndarray:
    design_rng = np.random.default_rng(2026)
    n_atoms = 14
    centers = design_rng.uniform(0.08, 0.92, size=(n_atoms, 2))
    directions = design_rng.normal(size=(n_atoms, 2))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    frequencies = np.linspace(1.5, 8.0, n_atoms)
    widths = design_rng.uniform(0.12, 0.28, size=n_atoms)
    phases = design_rng.uniform(0.0, 2.0 * np.pi, size=n_atoms)
    amplitudes = design_rng.uniform(0.7, 1.3, size=n_atoms)
    field = np.zeros(len(coordinates), dtype=float)
    effective_mass = np.zeros(len(coordinates), dtype=float)
    for center, direction, frequency, width, phase, amplitude in zip(
        centers, directions, frequencies, widths, phases, amplitudes
    ):
        offset = coordinates - center
        envelope = np.exp(-np.sum(offset * offset, axis=1) / (2.0 * width * width))
        projection = offset[:, 0] * direction[0] + offset[:, 1] * direction[1]
        field += amplitude * envelope * np.sin(
            2.0 * np.pi * frequency * projection + phase
        )
        effective_mass += envelope
    field /= np.sqrt(np.maximum(effective_mass, 0.25))
    return (field - field.mean()) / field.std()


def synthetic_sample(n: int = 1000, seed: int = 1042):
    rng = np.random.default_rng(seed)
    coordinates = rng.uniform(0.0, 1.0, size=(n, 2))
    s1, s2 = coordinates[:, 0], coordinates[:, 1]
    x1 = 0.8 * s1 - 0.35 * s2 + 0.28 * rng.normal(size=n)
    x2 = np.exp(-10.0 * ((s1 - 0.35) ** 2 + (s2 - 0.68) ** 2)) + 0.18 * rng.normal(size=n)
    x3 = np.sin(2.0 * np.pi * s1) * np.cos(2.0 * np.pi * s2) + 0.18 * rng.normal(size=n)
    trend = 1.20 * np.sin(np.pi * x1 * x3) + 0.85 * np.tanh(2.0 * x2 - 0.4) + 0.55 * x1**2 - 0.45 * x2 * x3
    spectrum = local_spectral_field(coordinates)
    noise_scale = 0.08 * (0.65 + 0.7 * s1)
    response = trend + 1.15 * spectrum + noise_scale * rng.normal(size=n)
    return coordinates, response, spectrum, noise_scale


def make_synthetic_figure() -> None:
    grid_axis = np.linspace(0, 1, 170)
    gx, gy = np.meshgrid(grid_axis, grid_axis)
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])
    grid_field = local_spectral_field(grid_points).reshape(gx.shape)
    coord, response, spectrum, noise_scale = synthetic_sample()

    pd.DataFrame(
        {
            "s1": coord[:, 0],
            "s2": coord[:, 1],
            "response": response,
            "latent_spatial_component": spectrum,
            "noise_sd": noise_scale,
        }
    ).to_csv(SOURCE_OUT / "synthetic_visualization_seed1042.csv", index=False)

    fig = plt.figure(figsize=(7.2, 2.45), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.0, 1.0], wspace=0.16)
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    surface = ax3d.plot_surface(
        gx,
        gy,
        grid_field,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        rcount=100,
        ccount=100,
    )
    ax3d.view_init(elev=31, azim=-58)
    ax3d.set_box_aspect((1.0, 1.0, 0.62))
    ax3d.set_xlabel(r"$s_1$", labelpad=-2)
    ax3d.set_ylabel(r"$s_2$", labelpad=-2)
    ax3d.set_zlabel("Value", labelpad=-1)
    ax3d.set_title("Localized Fourier component", pad=-1, fontweight="bold")
    ax3d.xaxis.pane.set_alpha(0.0)
    ax3d.yaxis.pane.set_alpha(0.0)
    ax3d.zaxis.pane.set_alpha(0.0)
    ax3d.grid(False)
    panel_label(ax3d, "a")
    ax = fig.add_subplot(gs[0, 1])
    image = ax.imshow(
        grid_field,
        origin="lower",
        extent=(0, 1, 0, 1),
        cmap="viridis",
        aspect="equal",
    )
    ax.set_title("Spatially varying latent field", fontweight="bold")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    panel_label(ax, "b")
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Standardized value")

    ax = fig.add_subplot(gs[0, 2])
    lo, hi = np.quantile(response, [0.01, 0.99])
    points = ax.scatter(
        coord[:, 0],
        coord[:, 1],
        c=np.clip(response, lo, hi),
        cmap="viridis",
        vmin=lo,
        vmax=hi,
        s=8,
        linewidths=0,
        alpha=0.92,
        rasterized=True,
    )
    ax.set_title("Sampled response ($n=1{,}000$)", fontweight="bold")
    ax.set_xlabel(r"$s_1$")
    ax.set_ylabel(r"$s_2$")
    ax.set_aspect("equal")
    panel_label(ax, "c")
    cbar = fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Observed response")
    export(fig, "synthetic_field_visualization_epoch300")


def load_lucas_large() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = DATA / "LUCAS 2018 TopSoil Dataset" / "OriginalSoilDataset.csv"
    frame = pd.read_csv(path, low_memory=False)
    coords = ["TH_LONG", "TH_LAT"]
    features = ["pH_CaCl2", "pH_H2O", "EC", "N", "P", "K", "Elev"]
    required = coords + features + ["OC"]
    frame = frame[required].apply(pd.to_numeric, errors="coerce").dropna()
    return (
        frame[coords].to_numpy(),
        frame[features].to_numpy(),
        np.log1p(frame["OC"].to_numpy()),
    )


def fit_lucas_histgbr(seed: int, expected_test_index: np.ndarray):
    coordinates, features, response = load_lucas_large()
    indices = np.arange(len(response))
    train_index, test_index = train_test_split(indices, test_size=0.2, random_state=seed)
    train_index, validation_index = train_test_split(
        train_index, test_size=0.1 / 0.8, random_state=seed
    )
    if not np.array_equal(test_index, expected_test_index):
        raise RuntimeError("LUCAS test split does not match the stored ASDKL prediction file")
    coordinate_scaler = StandardScaler().fit(coordinates[train_index])
    feature_scaler = StandardScaler().fit(features[train_index])
    response_scaler = StandardScaler().fit(response[train_index, None])
    design = np.column_stack(
        [coordinate_scaler.transform(coordinates), feature_scaler.transform(features)]
    )
    standardized_response = response_scaler.transform(response[:, None]).ravel()
    model = HistGradientBoostingRegressor(
        max_iter=300,
        l2_regularization=1e-3,
        random_state=seed,
    )
    model.fit(design[train_index], standardized_response[train_index])
    prediction = response_scaler.inverse_transform(
        model.predict(design[test_index])[:, None]
    ).ravel()
    return coordinates[test_index], response[test_index], prediction


def metric_text(target: np.ndarray, prediction: np.ndarray) -> str:
    rmse = mean_squared_error(target, prediction) ** 0.5
    mae = mean_absolute_error(target, prediction)
    r2 = r2_score(target, prediction)
    return f"RMSE={rmse:.3f}; MAE={mae:.3f}; $R^2$={r2:.3f}"


def style_spatial_axis(ax, longitude: bool = True) -> None:
    ax.set_xlabel("Longitude (°E)" if longitude else r"$s_1$")
    ax.set_ylabel("Latitude (°N)" if longitude else r"$s_2$")
    ax.grid(color=GRID, linewidth=0.45, zorder=0)
    ax.set_facecolor("#FBFCFD")


def make_lucas_comparison() -> None:
    seed = 1042
    stored = np.load(
        RESULTS / "lucas_large" / f"seed_{seed}" / "lucas_large_predictions.npz"
    )
    coordinates, target, hist = fit_lucas_histgbr(seed, stored["test_index"])
    if not np.allclose(coordinates, stored["coord"]):
        raise RuntimeError("LUCAS coordinates do not match the stored ASDKL source")
    if not np.allclose(target, stored["target"], atol=2e-6):
        raise RuntimeError("LUCAS target values do not match the stored ASDKL source")
    asdkl = stored["prediction"].astype(float)
    residual_asdkl = target - asdkl
    residual_hist = target - hist
    lo, hi = np.quantile(np.concatenate([target, asdkl, hist]), [0.01, 0.99])
    residual_limit = np.quantile(
        np.abs(np.concatenate([residual_asdkl, residual_hist])), 0.99
    )

    source = pd.DataFrame(
        {
            "test_index": stored["test_index"],
            "longitude": coordinates[:, 0],
            "latitude": coordinates[:, 1],
            "observed_log1p_OC": target,
            "ASDKL_prediction": asdkl,
            "ASDKL_residual_observed_minus_prediction": residual_asdkl,
            "HistGBR_prediction": hist,
            "HistGBR_residual_observed_minus_prediction": residual_hist,
        }
    )
    source.to_csv(SOURCE_OUT / "lucas_full_spatial_comparison_seed1042.csv", index=False)

    fig = plt.figure(figsize=(7.2, 4.05), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1.24, 1.0, 1.0, 0.055])
    axes = [
        fig.add_subplot(gs[:, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    cax_prediction = fig.add_subplot(gs[0, 3])
    cax_residual = fig.add_subplot(gs[1, 3])
    prediction_panels = [
        (axes[0], target, "Observed test responses", 5.0),
        (axes[1], asdkl, "ASDKL prediction\n" + metric_text(target, asdkl), 3.5),
        (axes[2], hist, "HistGBR prediction\n" + metric_text(target, hist), 3.5),
    ]
    prediction_scatter = None
    for ax, values, title, point_size in prediction_panels:
        prediction_scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=np.clip(values, lo, hi),
            cmap="viridis",
            vmin=lo,
            vmax=hi,
            s=point_size,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontweight="bold", linespacing=1.35)
        style_spatial_axis(ax)
    residual_norm = TwoSlopeNorm(
        vmin=-residual_limit, vcenter=0.0, vmax=residual_limit
    )
    residual_panels = [
        (axes[3], residual_asdkl, "ASDKL residual"),
        (axes[4], residual_hist, "HistGBR residual"),
    ]
    residual_scatter = None
    for ax, values, title in residual_panels:
        residual_scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=np.clip(values, -residual_limit, residual_limit),
            cmap="coolwarm",
            norm=residual_norm,
            s=3.5,
            alpha=0.88,
            linewidths=0,
            rasterized=True,
        )
        ax.set_title(title, fontweight="bold")
        style_spatial_axis(ax)
    for index, ax in enumerate(axes):
        panel_label(ax, chr(ord("a") + index))
    cbar = fig.colorbar(prediction_scatter, cax=cax_prediction)
    cbar.set_label(r"$log(1+\mathrm{OC})$")
    cbar = fig.colorbar(residual_scatter, cax=cax_residual)
    cbar.set_label("Residual (observed $-$ predicted)")
    export(fig, "lucas_full_asdkl_histgbr_comparison_epoch300")


def make_california_visualization() -> None:
    seed = 1042
    stored = np.load(
        RESULTS / "california" / f"seed_{seed}" / "california_predictions.npz"
    )
    coordinates = stored["coord"]
    target = stored["target"].astype(float)
    prediction = stored["prediction"].astype(float)
    residual = target - prediction
    lo, hi = np.quantile(np.concatenate([target, prediction]), [0.01, 0.99])
    residual_limit = np.quantile(np.abs(residual), 0.99)
    pd.DataFrame(
        {
            "test_index": stored["test_index"],
            "longitude": coordinates[:, 0],
            "latitude": coordinates[:, 1],
            "observed_median_house_value": target,
            "ASDKL_prediction": prediction,
            "residual_observed_minus_prediction": residual,
            "ASDKL_predictive_sd": stored["predictive_std"],
        }
    ).to_csv(SOURCE_OUT / "california_full_spatial_visualization_seed1042.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.55), constrained_layout=True)
    common = [
        (target, "Observed", "viridis", None, lo, hi),
        (prediction, "ASDKL\n" + metric_text(target, prediction), "viridis", None, lo, hi),
        (
            residual,
            "Residual (observed $-$ prediction)",
            "coolwarm",
            TwoSlopeNorm(vmin=-residual_limit, vcenter=0.0, vmax=residual_limit),
            None,
            None,
        ),
    ]
    scatters = []
    for index, (ax, (values, title, cmap, norm, vmin, vmax)) in enumerate(zip(axes, common)):
        kwargs = {"cmap": cmap, "norm": norm} if norm is not None else {"cmap": cmap, "vmin": vmin, "vmax": vmax}
        shown = np.clip(values, -residual_limit, residual_limit) if norm is not None else np.clip(values, lo, hi)
        scatter = ax.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=shown,
            s=4.2,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
            **kwargs,
        )
        scatters.append(scatter)
        ax.set_title(title, fontweight="bold", linespacing=1.35)
        style_spatial_axis(ax)
        panel_label(ax, chr(ord("a") + index))
    for scatter, ax in zip(scatters[:2], axes[:2]):
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label("Median house value ($100{,}000$)")
    cbar = fig.colorbar(scatters[2], ax=axes[2], fraction=0.046, pad=0.025)
    cbar.set_label("Residual")
    export(fig, "california_full_spatial_visualization_epoch300")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure", choices=["all", "synthetic", "lucas", "california"], default="all")
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    DATA = args.data_root
    if args.figure in ("all", "synthetic"):
        make_synthetic_figure()
    if args.figure in ("all", "lucas"):
        make_lucas_comparison()
    if args.figure in ("all", "california"):
        make_california_visualization()
    # The final six-block ablation figure is generated by summarize_revision14.py.
    print(f"Figures: {FIGURE_OUT}")
    print(f"Source data: {SOURCE_OUT}")
