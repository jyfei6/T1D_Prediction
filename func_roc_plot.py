import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def roc_plot(roc_files, colors):
    # 每条曲线的颜色（可按自己喜好改）

    plt.figure(figsize=(8, 6))

    for label, path in roc_files.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skip.")
            continue

        df = pd.read_csv(path)

        # 这里假设列名为 fpr, tpr_mean, tpr_ci95, mean_auc
        fpr      = df["fpr"].values
        tpr_mean = df["tpr_mean"].values
        ci95     = df["tpr_ci95"].values if "tpr_ci95" in df.columns else None

        if "mean_auc" in df.columns:
            mean_auc = float(df["mean_auc"].iloc[0])
            curve_label = f"{label} (AUC = {mean_auc:.2f})"
        else:
            curve_label = label

        c = colors.get(label, None)

        # 平均 ROC 曲线
        plt.plot(fpr, tpr_mean, lw=2, color=c, label=curve_label)

        # 95% CI 阴影（如果有）
        if ci95 is not None:
            lower = np.maximum(tpr_mean - ci95, 0)
            upper = np.minimum(tpr_mean + ci95, 1)
            plt.fill_between(fpr, lower, upper, color=c, alpha=0.15)

    # 随机猜测对角线
    plt.plot([0, 1], [0, 1], color="black", lw=1, linestyle="--")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate", fontsize=16)
    plt.ylabel("True Positive Rate", fontsize=16)
    plt.title("Average ROC Curves", fontsize=18, pad=20)
    plt.legend(loc="lower right")

    # 去掉上、右边框
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)

    plt.tight_layout()
    plt.show()
    return 0
