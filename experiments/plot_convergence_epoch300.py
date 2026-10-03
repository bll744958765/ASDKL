"""Plot 300-epoch ASDKL training and validation trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


RESULTS = Path(__file__).resolve().parents[1] / "results" / "revision14_shared_relation" / "six_block_reference"
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_figures"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (1042, 1052, 1062, 1072, 1082)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_history(seed: int) -> tuple[dict, int]:
    path = RESULTS / "synthetic" / f"seed_{seed}" / "synthetic_run.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    return record["training_history"], int(record["best_epoch"])


histories_and_epochs = [read_history(seed) for seed in SEEDS]
histories = [item[0] for item in histories_and_epochs]
best_epochs = np.asarray([item[1] for item in histories_and_epochs], dtype=float)
panels = [
    ("total", "Training objective", False),
    ("nll", "Training Gaussian NLL", False),
    ("mse", "Training MSE (diagnostic only)", True),
    ("val_total", "Validation objective", False),
    ("val_mse", "Validation MSE", True),
    ("spectral", "Weighted spectral penalty", True),
]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
for panel_index, (ax, (key, title, nonnegative)) in enumerate(zip(axes.flat, panels)):
    values = np.asarray([history[key] for history in histories], dtype=float)
    if key == "spectral":
        values *= 0.1
    epoch = np.asarray(histories[0]["epoch"], dtype=float)
    mean = np.nanmean(values, axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    interval = norm.ppf(0.975) * se
    lower = np.maximum(0, mean - interval) if nonnegative else mean - interval
    ax.plot(epoch, mean, color="#B6493A", lw=1.25, label="Five-seed mean")
    ax.fill_between(epoch, lower, mean + interval, color="#E9A99F", alpha=0.42, label="95% CI")
    if key in {"val_total", "val_mse"}:
        median_epoch = float(np.median(best_epochs))
        ax.axvline(median_epoch, color="#236987", ls="--", lw=0.9,
                   label=f"Median selected epoch ({median_epoch:.0f})")
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(color="#E6EAED", lw=0.5)
    ax.text(-0.12, 1.06, chr(97 + panel_index), transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="top")
axes[0, 0].legend(frameon=False, fontsize=6.2)
axes[1, 1].legend(frameon=False, fontsize=6.2)

for extension in ("pdf", "svg", "png", "tiff"):
    fig.savefig(
        OUT / f"synthetic_convergence_epoch300.{extension}",
        dpi=600 if extension == "tiff" else 300,
        bbox_inches="tight",
        facecolor="white",
    )
plt.close(fig)
print("best_epochs", best_epochs.astype(int).tolist())
print("output", OUT)
