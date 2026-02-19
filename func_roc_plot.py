import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def roc_plot(roc_files, colors, out_pdf=None, dpi=300):
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(12, 8))

    for label, path in roc_files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skip.")
            continue

        df = pd.read_csv(path)

        fpr      = df["fpr"].values
        tpr_mean = df["tpr_mean"].values
        ci95     = df["tpr_ci95"].values if "tpr_ci95" in df.columns else None

        if "mean_auc" in df.columns:
            mean_auc = float(df["mean_auc"].iloc[0])
            curve_label = f"{label} (AUC = {mean_auc:.2f})"
        else:
            curve_label = label

        c = colors.get(label, None)

        ax.plot(fpr, tpr_mean, lw=2, color=c, label=curve_label)

        if ci95 is not None:
            lower = np.maximum(tpr_mean - ci95, 0)
            upper = np.minimum(tpr_mean + ci95, 1)
            ax.fill_between(fpr, lower, upper, color=c, alpha=0.15)

    ax.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    ax.set_title("Average ROC Curves", fontsize=18, pad=20)
    ax.legend(loc="lower right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=18)
    ax.tick_params(axis="y", labelsize=18)

    fig.tight_layout()

    if out_pdf is not None:
        fig.savefig(out_pdf, bbox_inches="tight", dpi=dpi)

    plt.show()
    return 0
