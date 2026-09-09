"""Generate the README preview of the nonstationary synthetic benchmark."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "common"))

from synthetic_dataset import SyntheticSpatialDataset


def main() -> None:
    coordinates, covariates, response = SyntheticSpatialDataset(
        n_samples=3000, random_state=1042
    ).generate()
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    panels = [(covariates[:, 0], "Nonlinear covariate"), (response, "Synthetic response")]
    for label, ax, (values, title) in zip("ab", axes, panels):
        scatter = ax.scatter(
            coordinates[:, 0], coordinates[:, 1], c=values, cmap="viridis",
            s=5, alpha=0.9, linewidths=0, rasterized=True,
        )
        ax.set(title=title, xlabel="$s_1$", ylabel="$s_2$")
        ax.set_aspect("equal")
        ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontweight="bold")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.025)
    output = REPOSITORY_ROOT / "docs" / "synthetic_example.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
