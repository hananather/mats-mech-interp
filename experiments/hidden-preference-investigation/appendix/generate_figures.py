#!/usr/bin/env python3
"""Generate figures for the hidden belief experiment report.

Uses hardcoded data from 18 experiment runs. No notebook parsing required.
Outputs 4 PNG charts to the figures/ subdirectory.
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Shared style config
BEHAVIORAL_COLOR = "#E8836B"  # coral/salmon
SAE_COLOR = "#4682B4"  # steelblue
TITLE_SIZE = 14
LABEL_SIZE = 12
TICK_SIZE = 11
DPI = 150

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    # Older matplotlib versions use different style names
    try:
        plt.style.use("seaborn-whitegrid")
    except OSError:
        pass  # Fall back to default


# ---------------------------------------------------------------------------
# Data: v2 Hidden Belief Runs (matched prompts)
# ---------------------------------------------------------------------------
v2_behavioral_cells = [35, 24, 33]
v2_behavioral_errors = [3, 1, 2]
v2_sae_cells = [43, 48, 21]
v2_sae_errors = [4, 5, 3]

# ---------------------------------------------------------------------------
# Data: False Positive Controls
# ---------------------------------------------------------------------------
# Behavioral: 3/3 false positives (all incorrect)
# SAE: 1/3 false positives among complete runs (fp-sae-run2 correct, fp-sae-run3 correct,
#       fp-sae-run4 false positive; fp-sae-run1 incomplete, excluded)
fp_behavioral_rate = 100.0  # 3/3
fp_sae_rate = 33.3  # 1/3
fp_behavioral_n = "3/3"
fp_sae_n = "1/3"

# ---------------------------------------------------------------------------
# Data: SAE Feature Activations (from v2-sae-run3)
# ---------------------------------------------------------------------------
features = [
    ("Gender/sex differences", 16085, 46.936),
    ("Women's achievements", 5629, 34.943),
    ("Women's relationships", 16305, 20.235),
    ("Menstrual health", 15675, 16.725),
    ("Female characters", 6259, 16.129),
]
feature_labels = [f[0] for f in features]
feature_ids = [f[1] for f in features]
feature_values = [f[2] for f in features]


# ---------------------------------------------------------------------------
# Chart 1: False Positive Rate by Condition
# ---------------------------------------------------------------------------
def generate_fp_rate():
    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["Behavioral", "SAE"]
    rates = [fp_behavioral_rate, fp_sae_rate]
    colors = [BEHAVIORAL_COLOR, SAE_COLOR]
    annotations = [fp_behavioral_n, fp_sae_n]

    bars = ax.bar(conditions, rates, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, annotation in zip(bars, annotations):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 2,
            annotation,
            ha="center",
            va="bottom",
            fontsize=LABEL_SIZE,
            fontweight="bold",
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("False Positive Rate (%)", fontsize=LABEL_SIZE)
    ax.set_title(
        "False Positive Rate by Condition",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        pad=20,
    )
    ax.text(
        0.5,
        1.02,
        "N=3 complete runs per condition",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="gray",
    )
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.set_yticks([0, 25, 50, 75, 100])

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fp_rate.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'fp_rate.png'}")


# ---------------------------------------------------------------------------
# Chart 2: Cell Count Distribution
# ---------------------------------------------------------------------------
def generate_cell_counts():
    fig, ax = plt.subplots(figsize=(8, 5))

    run_labels = ["Run 1", "Run 2", "Run 3"]
    x = np.arange(len(run_labels))
    width = 0.32

    bars_b = ax.bar(x - width / 2, v2_behavioral_cells, width, label="Behavioral", color=BEHAVIORAL_COLOR, edgecolor="white", linewidth=1.2)
    bars_s = ax.bar(x + width / 2, v2_sae_cells, width, label="SAE", color=SAE_COLOR, edgecolor="white", linewidth=1.2)

    # Add value labels on bars
    for bar in list(bars_b) + list(bars_s):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Mean lines
    behavioral_mean = np.mean(v2_behavioral_cells)
    sae_mean = np.mean(v2_sae_cells)
    ax.axhline(y=behavioral_mean, color=BEHAVIORAL_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Behavioral mean: {behavioral_mean:.1f}")
    ax.axhline(y=sae_mean, color=SAE_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"SAE mean: {sae_mean:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Number of Cells", fontsize=LABEL_SIZE)
    ax.set_title("Investigation Length (v2 Matched-Prompt Runs)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, max(max(v2_behavioral_cells), max(v2_sae_cells)) + 10)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cell_counts.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'cell_counts.png'}")


# ---------------------------------------------------------------------------
# Chart 3: Error Count by Run
# ---------------------------------------------------------------------------
def generate_error_rates():
    fig, ax = plt.subplots(figsize=(8, 5))

    run_labels = ["Run 1", "Run 2", "Run 3"]
    x = np.arange(len(run_labels))
    width = 0.32

    bars_b = ax.bar(x - width / 2, v2_behavioral_errors, width, label="Behavioral", color=BEHAVIORAL_COLOR, edgecolor="white", linewidth=1.2)
    bars_s = ax.bar(x + width / 2, v2_sae_errors, width, label="SAE", color=SAE_COLOR, edgecolor="white", linewidth=1.2)

    # Add value labels on bars
    for bar in list(bars_b) + list(bars_s):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            str(int(height)),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Mean lines
    behavioral_mean = np.mean(v2_behavioral_errors)
    sae_mean = np.mean(v2_sae_errors)
    ax.axhline(y=behavioral_mean, color=BEHAVIORAL_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"Behavioral mean: {behavioral_mean:.1f}")
    ax.axhline(y=sae_mean, color=SAE_COLOR, linestyle="--", linewidth=1.5, alpha=0.7,
               label=f"SAE mean: {sae_mean:.1f}")

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Number of Errors", fontsize=LABEL_SIZE)
    ax.set_title("Error Count (v2 Matched-Prompt Runs)", fontsize=TITLE_SIZE, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_ylim(0, max(max(v2_behavioral_errors), max(v2_sae_errors)) + 2)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_rates.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'error_rates.png'}")


# ---------------------------------------------------------------------------
# Chart 4: SAE Feature Activations (horizontal bar chart)
# ---------------------------------------------------------------------------
def generate_feature_heatmap():
    fig, ax = plt.subplots(figsize=(8, 4))

    # Reverse order so highest activation is at top
    labels_rev = feature_labels[::-1]
    values_rev = feature_values[::-1]
    ids_rev = feature_ids[::-1]

    # Color gradient: light to dark blue based on value
    max_val = max(feature_values)
    norm_values = [v / max_val for v in values_rev]
    colors = [plt.cm.Blues(0.3 + 0.6 * nv) for nv in norm_values]

    y_pos = np.arange(len(labels_rev))
    bars = ax.barh(y_pos, values_rev, color=colors, edgecolor="white", linewidth=1.2, height=0.6)

    # Add value labels on each bar
    for bar, val, fid in zip(bars, values_rev, ids_rev):
        w = bar.get_width()
        ax.text(
            w + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            ha="left",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    # Y-axis labels with feature ID
    y_labels = [f"{lbl} (#{fid})" for lbl, fid in zip(labels_rev, ids_rev)]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=TICK_SIZE)
    ax.set_xlabel("Activation (fine-tuned model)", fontsize=LABEL_SIZE)
    ax.set_title("SAE Features Activated by Gender Belief Adapter", fontsize=TITLE_SIZE, fontweight="bold")
    ax.set_xlim(0, max(feature_values) + 8)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_heatmap.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {FIGURES_DIR / 'feature_heatmap.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating figures...")
    generate_fp_rate()
    generate_cell_counts()
    generate_error_rates()
    generate_feature_heatmap()
    print("Done. All figures saved to:", FIGURES_DIR.resolve())
