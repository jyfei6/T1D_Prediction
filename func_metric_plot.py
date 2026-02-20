import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def metric_boxplot(
    metric_files: dict,              # {"Sample1": "a.csv", "Sample2": "b.csv", ...}
    palette_boxes: dict,             # {"Sample1": "#d62728", "Sample2": "#1f77b4", ...}
    cols=("Accuracy", "Specificity", "Sensitivity"),
    title="Performance Comparison",
    out_pdf=None,                    # e.g. "figure_performance_metrics.pdf"
    ylim=(0, 1.05),
    figsize=(16, 10),
    showfliers=False,
    width=0.8,
    linewidth=1.5,
    median_lw=2.8,
    jitter=0.2,
    point_size=3,
    point_alpha=0.6,
    legend_loc="lower left",
):
    """
    仅做你给的那种可视化风格：boxplot + black strip dots, 厚边框/中位线, 去重 legend, 去 grid, 旋转x标签。
    CSV 列名默认固定为 cols。
    """

    # --- read & prep: long-form ---
    longs = []
    for exp, fp in metric_files.items():
        df = pd.read_csv(fp)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])

        df = df[list(cols)].copy()
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(how="any")

        df["Experiment"] = exp
        long_df = df.melt(id_vars=["Experiment"], var_name="Metric", value_name="Value")
        longs.append(long_df)

    plot_df = pd.concat(longs, ignore_index=True).dropna(subset=["Metric", "Value"])
    plot_df = plot_df[plot_df["Metric"].isin(cols)]

    order = list(cols)
    hue_order = list(metric_files.keys())

    # --- plot ---
    sns.set_style("white")

    plt.figure(figsize=figsize)
    ax = sns.boxplot(
        data=plot_df, x="Metric", y="Value", hue="Experiment",
        order=order, hue_order=hue_order,
        showfliers=showfliers, width=width,
        palette={k: palette_boxes[k] for k in hue_order},
        linewidth=linewidth,
        medianprops={"linewidth": median_lw, "color": "black"},
    )

    sns.stripplot(
        data=plot_df, x="Metric", y="Value", hue="Experiment",
        order=order, hue_order=hue_order,
        dodge=True, jitter=jitter, size=point_size, alpha=point_alpha,
        color="black", edgecolor="none", zorder=10, ax=ax
    )

    # --- tidy legend (remove duplicates from double-plot) ---
    handles, labels = ax.get_legend_handles_labels()
    n = len(hue_order)
    ax.legend(handles[:n], labels[:n], title="Experiment",
              frameon=True, loc=legend_loc, title_fontsize=10, prop={"size": 10})

    # --- cosmetics ---
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_ylabel("Score", fontsize=26)
    ax.set_xlabel("Metric", fontsize=26)
    ax.set_title(title, fontsize=26)
    ax.tick_params(axis="both", labelsize=26)

    ax.grid(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    plt.tight_layout()
    if out_pdf:
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.show()

    return plot_df  # 方便你后续 describe / 统计
