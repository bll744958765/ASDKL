"""Regenerate the manuscript figures from the CSV files in ``figure/data``.

The script never reads model checkpoints or test responses outside this folder.
This makes visual changes independent of model training and keeps every plotted
number traceable to a compact source-data file.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DEFAULT_OUTPUT = HERE / "output"
BLUE = "#2C6F93"
TEAL = "#65A695"
RED = "#B6493A"
GRID = "#E5EAEE"

mpl.rcParams.update({
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
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})


def panel_label(ax, label: str) -> None:
    method = ax.text2D if hasattr(ax, "text2D") else ax.text
    method(0.015, 0.985, label, transform=ax.transAxes, va="top",
           fontweight="bold", fontsize=8,
           bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 0.4})


def export(fig, output: Path, stem: str, formats: list[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        dpi = 600 if extension == "tiff" else 300
        fig.savefig(output / f"{stem}.{extension}", dpi=dpi,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def synthetic_field(output: Path, formats: list[str]) -> None:
    frame = pd.read_csv(DATA / "synthetic_visualization_seed1042.csv")
    fig = plt.figure(figsize=(7.2, 2.45), constrained_layout=True)
    ax3d = fig.add_subplot(1, 3, 1, projection="3d")
    surf = ax3d.plot_trisurf(frame.s1, frame.s2, frame.latent_spatial_component,
                            cmap="viridis", linewidth=0, antialiased=True)
    ax3d.set_title("Latent spatial component", fontweight="bold")
    ax3d.set_xlabel(r"$s_1$"); ax3d.set_ylabel(r"$s_2$"); ax3d.set_zlabel("Value")
    panel_label(ax3d, "a")
    ax = fig.add_subplot(1, 3, 2)
    points = ax.scatter(frame.s1, frame.s2, c=frame.latent_spatial_component,
                        cmap="viridis", s=8, linewidth=0)
    ax.set_title("Localized spatial regimes", fontweight="bold")
    ax.set_xlabel(r"$s_1$"); ax.set_ylabel(r"$s_2$"); panel_label(ax, "b")
    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03, label="Latent value")
    ax = fig.add_subplot(1, 3, 3)
    points = ax.scatter(frame.s1, frame.s2, c=frame.response,
                        cmap="viridis", s=8, linewidth=0)
    ax.set_title("Sampled observations", fontweight="bold")
    ax.set_xlabel(r"$s_1$"); ax.set_ylabel(r"$s_2$"); panel_label(ax, "c")
    fig.colorbar(points, ax=ax, fraction=0.046, pad=0.03, label="Response")
    export(fig, output, "synthetic_field_visualization", formats)


def lucas_comparison(output: Path, formats: list[str]) -> None:
    frame = pd.read_csv(DATA / "lucas_full_spatial_comparison_seed1042.csv")
    observed = frame.observed_log1p_OC.to_numpy()
    asdkl = frame.ASDKL_prediction.to_numpy()
    hist = frame.HistGBR_prediction.to_numpy()
    asdkl_residual = frame.ASDKL_residual_observed_minus_prediction.to_numpy()
    hist_residual = frame.HistGBR_residual_observed_minus_prediction.to_numpy()
    lo, hi = np.percentile(np.r_[observed, asdkl, hist], [1, 99])
    residual_limit = np.percentile(np.abs(np.r_[asdkl_residual, hist_residual]), 99)
    fig = plt.figure(figsize=(7.2, 4.55), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.08, 1, 1])
    axes = [fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[0, 2]), fig.add_subplot(grid[1, 1]),
            fig.add_subplot(grid[1, 2])]
    values = [observed, asdkl, hist, asdkl_residual, hist_residual]
    titles = ["Observed test responses", "ASDKL-LocalGP prediction",
              "HistGBR prediction", "ASDKL-LocalGP residual", "HistGBR residual"]
    for index, (ax, values_i, title) in enumerate(zip(axes, values, titles)):
        if index < 3:
            artist = ax.scatter(frame.longitude, frame.latitude, c=values_i,
                                cmap="viridis", vmin=lo, vmax=hi, s=4, linewidth=0)
        else:
            artist = ax.scatter(frame.longitude, frame.latitude, c=values_i,
                                cmap="coolwarm", norm=TwoSlopeNorm(0, -residual_limit, residual_limit),
                                s=4, linewidth=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.grid(color=GRID, linewidth=0.4); panel_label(ax, chr(97 + index))
    fig.colorbar(axes[2].collections[0], ax=axes[1:3], fraction=0.035, pad=0.02,
                 label=r"$\log(1+\mathrm{OC})$")
    fig.colorbar(axes[4].collections[0], ax=axes[3:5], fraction=0.035, pad=0.02,
                 label="Residual (observed − predicted)")
    export(fig, output, "lucas_full_asdkl_histgbr_comparison", formats)


def california_map(output: Path, formats: list[str]) -> None:
    frame = pd.read_csv(DATA / "california_full_spatial_visualization_seed1042.csv")
    observed = frame.observed_median_house_value.to_numpy()
    predicted = frame.ASDKL_prediction.to_numpy()
    residual = frame.residual_observed_minus_prediction.to_numpy()
    lo, hi = np.percentile(np.r_[observed, predicted], [1, 99])
    residual_limit = np.percentile(np.abs(residual), 99)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.35), constrained_layout=True)
    titles = ["Observed", "ASDKL-LocalGP", "Residual (observed − prediction)"]
    for index, (ax, values, title) in enumerate(zip(axes, [observed, predicted, residual], titles)):
        if index < 2:
            ax.scatter(frame.longitude, frame.latitude, c=values, cmap="viridis",
                       vmin=lo, vmax=hi, s=5, linewidth=0)
        else:
            ax.scatter(frame.longitude, frame.latitude, c=values, cmap="coolwarm",
                       norm=TwoSlopeNorm(0, -residual_limit, residual_limit), s=5, linewidth=0)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
        ax.grid(color=GRID, linewidth=0.4); panel_label(ax, chr(97 + index))
    for ax in axes[:2]:
        fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.025,
                     label="Median house value ($100,000)")
    fig.colorbar(axes[2].collections[0], ax=axes[2], fraction=0.046, pad=0.025,
                 label="Residual")
    export(fig, output, "california_full_spatial_visualization", formats)


def ablation(output: Path, formats: list[str]) -> None:
    source = pd.read_csv(DATA / "ablation_bar_source.csv")
    configurations = ["no_context", "no_fourier", "no_spectral_regularization",
                      "no_spectral_attention", "no_gp", "no_residual_connections",
                      "three_residual_blocks", "six_residual_blocks", "auxiliary_mse"]
    labels = ["Remove context encoder", "Remove adaptive Fourier features",
              "Remove spectral regularization", "Remove spectral attention",
              "Remove residual GP", "Remove residual connections",
              "Use three residual blocks", "Use six residual blocks", "Add auxiliary MSE"]
    y = np.array([0, 1, 2, 3, 4, 5, 6.55, 7.55, 9.1])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.25), constrained_layout=True, sharey=True)
    for panel, (ax, dataset, title, color, marker) in enumerate(zip(
            axes, ["synthetic", "pm25"], ["Nonstationary synthetic", r"EPA PM$_{2.5}$"],
            [BLUE, TEAL], ["o", "s"])):
        means, standard_deviations = [], []
        for position, configuration in zip(y, configurations):
            values = source.loc[(source.dataset == dataset) &
                                (source.configuration == configuration), "delta_rmse_pct"].to_numpy()
            means.append(values.mean()); standard_deviations.append(values.std(ddof=1))
            ax.scatter(values, position + np.linspace(-0.16, 0.16, len(values)), s=9,
                       marker=marker, facecolor="white", edgecolor=color, linewidth=0.5, zorder=3)
        ax.barh(y, means, xerr=standard_deviations, height=0.56, color=color,
                alpha=0.92, edgecolor="white", linewidth=0.45,
                error_kw={"ecolor": "#34434C", "elinewidth": 0.7, "capsize": 2}, zorder=2)
        ax.axvspan(-33, 0, color="#F2F6F4", zorder=0); ax.axvline(0, color="#36454F", lw=0.8, ls="--")
        ax.axhline(5.78, color="#D7DEE3", lw=0.7); ax.axhline(8.32, color="#D7DEE3", lw=0.7)
        ax.set_xlim(-33, 18); ax.set_ylim(-0.62, 9.72); ax.invert_yaxis()
        ax.set_xlabel("Change in test RMSE (%)"); ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", color=GRID, linewidth=0.5); panel_label(ax, chr(97 + panel))
    axes[0].set_yticks(y, labels); axes[1].tick_params(axis="y", labelleft=False)
    export(fig, output, "ablation", formats)


def convergence(output: Path, formats: list[str]) -> None:
    frame = pd.read_csv(DATA / "convergence_synthetic.csv")
    panels = [("total", "Training objective", False), ("nll", "Training Gaussian NLL", False),
              ("mse", "Training MSE (diagnostic only)", True),
              ("val_total", "Validation objective", False), ("val_mse", "Validation MSE", True),
              ("spectral", "Weighted spectral penalty", True)]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
    best_epochs = frame.groupby("seed").best_epoch.first().to_numpy()
    for index, (ax, (column, title, nonnegative)) in enumerate(zip(axes.flat, panels)):
        pivot = frame.pivot(index="seed", columns="epoch", values=column).sort_index(axis=1)
        values = pivot.to_numpy(float) * (0.1 if column == "spectral" else 1.0)
        epoch = pivot.columns.to_numpy(float); mean = values.mean(0)
        interval = 1.96 * values.std(0, ddof=1) / np.sqrt(values.shape[0])
        lower = np.maximum(0, mean - interval) if nonnegative else mean - interval
        ax.plot(epoch, mean, color=RED, lw=1.25, label="Five-seed mean")
        ax.fill_between(epoch, lower, mean + interval, color="#E9A99F", alpha=0.42, label="95% CI")
        if column in {"val_total", "val_mse"}:
            ax.axvline(np.median(best_epochs), color=BLUE, ls="--", lw=0.9,
                       label=f"Median selected epoch ({np.median(best_epochs):.0f})")
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(color=GRID, linewidth=0.5); panel_label(ax, chr(97 + index))
    axes[0, 0].legend(fontsize=6.2); axes[1, 1].legend(fontsize=6.2)
    export(fig, output, "synthetic_convergence", formats)


def sensitivity(output: Path, formats: list[str]) -> None:
    frame = pd.read_csv(DATA / "sensitivity_summary.csv")
    frame = frame[frame.dataset == "synthetic"]
    families = ["neighbors", "freq_dim", "hidden_dim", "lambda_spectral", "lr", "epochs"]
    titles = {"neighbors": r"Spatial neighbours $K$", "freq_dim": r"Frequency dimension $M$",
              "hidden_dim": "Representation width", "lambda_spectral": r"Spectral weight $\lambda_s$",
              "lr": "Learning rate", "epochs": "Training epochs"}
    ci_all = 4.303 * frame.relative_val_rmse_sd / np.sqrt(frame.n)
    ymin = float((frame.relative_val_rmse - ci_all).min() - 0.03)
    ymax = float((frame.relative_val_rmse + ci_all).max() + 0.03)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0), constrained_layout=True)
    for index, (ax, family) in enumerate(zip(axes.flat, families)):
        group = frame[frame.family == family].sort_values("value")
        values = group.relative_val_rmse.to_numpy(); ci = 4.303 * group.relative_val_rmse_sd / np.sqrt(group.n)
        positions = np.arange(len(group))
        ax.fill_between(positions, values-ci, values+ci, color=BLUE, alpha=0.16, linewidth=0)
        ax.plot(positions, values, marker="o", ms=3.5, lw=1.1, color=BLUE)
        ax.set_xticks(positions, [f"{value:g}" for value in group.value])
        ax.axhline(1, color="#666666", ls="--", lw=0.7); ax.set_ylim(ymin, ymax)
        ax.set_title(titles[family]); ax.set_xlabel("Value"); ax.set_ylabel("Validation RMSE / default")
        ax.grid(axis="y", color=GRID, linewidth=0.5); panel_label(ax, chr(97 + index))
    export(fig, output, "hyperparameter_sensitivity_synthetic", formats)


FIGURES = {
    "synthetic": synthetic_field,
    "lucas": lucas_comparison,
    "california": california_map,
    "ablation": ablation,
    "convergence": convergence,
    "sensitivity": sensitivity,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate ASDKL manuscript figures from source CSV files.")
    parser.add_argument("--figure", choices=["all", *FIGURES], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", nargs="+", choices=["svg", "pdf", "png", "tiff"],
                        default=["svg", "pdf", "png"])
    args = parser.parse_args()
    selected = FIGURES.items() if args.figure == "all" else [(args.figure, FIGURES[args.figure])]
    for name, function in selected:
        function(args.output_dir, args.formats)
        print(f"created {name}")


if __name__ == "__main__":
    main()
