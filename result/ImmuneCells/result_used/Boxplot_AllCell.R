# ==========================
# 0. 加载依赖包
# ==========================
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)

# ==========================
# 1. 配置：文件名和组名映射
# ==========================

file_map <- c(
  "Scaffold"    = "Scaffold.csv",
  "B Naive"     = "BNaive.csv",
  "CD4 EM"      = "CD4EM.csv",
  "CD8 Naive"   = "CD8Naive.csv",
  "C Monocyte"  = "CMonocyte.csv",
  "NK"          = "NK.csv",
  "Treg"        = "Treg.csv"
)

base_dir <- "/Users/JiangYIfei/Desktop/college/UM/lab/New_prediction_scaffold/result/blood_cells_res/"

# ==========================
# 2. 读取所有 csv，拼成 long-format DataFrame
# ==========================

all_list <- list()

for (group_name in names(file_map)) {
  fname <- file_map[[group_name]]
  fpath <- file.path(base_dir, fname)
  
  # 对应 Python: pd.read_csv(fpath, index_col=0)
  df <- read_csv(fpath)
  # 如果第一列是 index，需要丢掉，可以用：
  # df <- read_csv(fpath) %>% select(-1)
  
  # 确保有这三列
  needed_cols <- c("Accuracy", "Specificity", "Sensitivity")
  if (!all(needed_cols %in% colnames(df))) {
    stop(paste0(fname, " 中没有列: ", 
                paste(setdiff(needed_cols, colnames(df)), collapse = ", ")))
  }
  
  # 只保留三列，并转为 long format
  df_long <- df %>%
    select(all_of(needed_cols)) %>%
    pivot_longer(
      cols = everything(),
      names_to = "Metric",
      values_to = "Value"
    ) %>%
    filter(!is.na(Value)) %>%
    mutate(Group = group_name)
  
  all_list[[group_name]] <- df_long
}

plot_df <- bind_rows(all_list)
# 控制 Group 的顺序：按照 file_map 的名字顺序
plot_df <- plot_df %>%
  mutate(
    Group = factor(Group, levels = names(file_map))
  )

# 再继续你原来的 Metric 设置
metric_order <- c("Accuracy", "Specificity", "Sensitivity")
plot_df <- plot_df %>%
  mutate(
    Metric = factor(Metric, levels = metric_order)
  ) %>%
  mutate(
    Metric_plot = factor(
      as.character(Metric),
      levels = c(
        "Accuracy",
        "Accuracy_Spacer",
        "Specificity",
        "Specificity_Spacer",
        "Sensitivity"
      )
    )
  )

library(ggprism)

p <- ggplot(plot_df, aes(x = Metric_plot, y = Value, fill = Group)) +
  geom_boxplot(
    position = position_dodge2(width = 0.9, padding = 0.3),
    width = 0.8,
    outlier.size = 0.7,
    linewidth = 0.6
  ) +
  scale_x_discrete(
    labels = c(
      "Accuracy"           = "Accuracy",
      "Accuracy_Spacer"    = "",
      "Specificity"        = "Specificity",
      "Specificity_Spacer" = "",
      "Sensitivity"        = "Sensitivity"
    )
  ) +
  labs(
    x = "Metric",
    y = "Score",
    title = "SVC Performance by Metric and Cell Type",
    fill = "Cell type"
  ) +
  scale_y_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.1),
    expand = expansion(mult = c(0.02, 0.05))
  ) +
  # 这里加 limits，顺序就是 Scaffold, B Naive, CD4 EM, ...
  scale_fill_prism(
    palette = "floral",
    limits = names(file_map)
  ) +
  theme_bw(base_size = 16) +
  theme(
    panel.grid = element_blank(),
    axis.title = element_text(face = "bold"),
    plot.title = element_text(
      hjust = 0.5,
      face = "bold",
      margin = margin(b = 10)
    ),
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    legend.key.size = unit(0.9, "lines")
  )

print(p)

ggsave("svc_performance_boxplot_combined_ggplot.png", p,
       width = 10, height = 5, dpi = 300)
ggsave("svc_performance_boxplot_combined_ggplot.pdf", p,
       width = 10, height = 5)

