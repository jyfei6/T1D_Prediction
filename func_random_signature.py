# svc_random100_cv.py

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    roc_auc_score,
)


def random_sel(data, metadata, runSetting, param_grid):
    # 仅保留有标签的样本
    meta_labeled = metadata[~metadata["Group"].isna()].copy()
    sample_cols = [s for s in meta_labeled["Samples"] if s in data.columns]

    X_full = data.loc[:, sample_cols].T  # shape: n_samples x n_genes
    y_raw  = meta_labeled.set_index("Samples").loc[sample_cols, "Group"]
    y = LabelEncoder().fit_transform(y_raw)  # 二分类编码（0/1）
    if runSetting is None:
        N_RUNS = 1000
        N_FEATURES_PER_RUN = 100
        TEST_SIZE = 0.30
        RANDOM_SEED = 42
    else:
        N_RUNS = runSetting.N_RUNS
        N_FEATURES_PER_RUN = runSetting.N_FEATURES_PER_RUN
        TEST_SIZE = runSetting.TEST_SIZE
        RANDOM_SEED = runSetting.RANDOM_SEED

    if param_grid is None:
        param_grid = [
            {"kernel": ["linear"], "C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
            {
                "kernel": ["poly"],
                "degree": [2, 3, 4, 5],
                "gamma": ["scale", "auto"],
                "C": [0.1, 1, 10],
            },
        ]

    # ==========================
    # 3. 指标与 ROC 累积器
    # ==========================
    acc_list  = []
    sens_list = []
    spec_list = []
    auc_list  = []

    # 每次 run 的 fpr / tpr（用于之后统一插值）
    fpr_list = []
    tpr_list = []

    # 随机数发生器（用于基因子集选择）
    rng = np.random.default_rng(RANDOM_SEED)
    all_genes = np.array(data.index)  # 全部基因名（行名）

    # ==========================
    # 4. 主循环：随机基因 + 内层CV + 外层测试
    # ==========================
    for run in range(N_RUNS):
        # 1) 随机抽取 N_FEATURES_PER_RUN 个基因
        chosen_genes = rng.choice(all_genes, size=N_FEATURES_PER_RUN, replace=False)
        X_sub = X_full.loc[:, chosen_genes].values  # ndarray (n_samples, N_FEATURES_PER_RUN)

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub,
            y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=RANDOM_SEED + run,
        )

        # 3) 内层交叉验证（5 折），选择最优超参数
        base_svc = SVC(probability=True, class_weight="balanced", max_iter=10000)
        inner_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_SEED + run,
        )

        grid = GridSearchCV(
            estimator=base_svc,
            param_grid=param_grid,
            scoring="accuracy",   # 如需以 AUC 选参，可改为 "roc_auc"
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
        grid.fit(X_tr, y_tr)
        best_model = grid.best_estimator_

        # 4) 在外层测试集上评估
        y_pred = best_model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)

        # 混淆矩阵：tn, fp, fn, tp
        tn, fp, fn, tp = confusion_matrix(y_te, y_pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity / recall of positive class
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity

        acc_list.append(acc)
        sens_list.append(sens)
        spec_list.append(spec)

        # ===== 计算该 run 的 ROC 曲线（基于外层 test 集） =====
        y_proba = best_model.predict_proba(X_te)[:, 1]  # 取阳性类概率
        fpr, tpr, _ = roc_curve(y_te, y_proba)
        auc = roc_auc_score(y_te, y_proba)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc)

    # ==========================
    # 5. 汇总报告
    # ==========================
    avg_acc  = float(np.mean(acc_list))
    avg_sens = float(np.mean(sens_list))
    avg_spec = float(np.mean(spec_list))
    avg_auc  = float(np.mean(auc_list))

    print("===== SVC  =====")
    print(f"N_RUNS:             {N_RUNS}")
    print(f"Average Accuracy:    {avg_acc:.4f}")
    print(f"Average Specificity: {avg_spec:.4f}")
    print(f"Average Sensitivity: {avg_sens:.4f}")
    print(f"Average AUC:         {avg_auc:.4f}")

    # ==========================
    # 6. 保存每次 run 的 summary 指标
    # ==========================
    run_metrics = pd.DataFrame({
        "Accuracy":    acc_list,
        "Specificity": spec_list,
        "Sensitivity": sens_list,
        "AUC":         auc_list,
    })

    # ==========================
    # 7. 计算平均 ROC + 95% CI，并保存为 CSV
    # ==========================
    # 1) 统一 FPR 网格
    mean_fpr = np.linspace(0, 1, 500)  # 500 点的均匀网格

    # 2) 对每条 ROC 插值
    tpr_interp_list = []
    for fpr, tpr in zip(fpr_list, tpr_list):
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp_list.append(tpr_interp)

    tpr_interp_arr = np.vstack(tpr_interp_list)  # shape: (N_RUNS, 500)

    # 3) 计算均值、标准差、95% CI
    tpr_mean = np.mean(tpr_interp_arr, axis=0)
    tpr_std  = np.std(tpr_interp_arr, axis=0)
    n        = tpr_interp_arr.shape[0]

    ci95 = 1.96 * tpr_std / np.sqrt(n)

    # 规范端点
    tpr_mean[0]  = 0.0
    tpr_mean[-1] = 1.0

    # 4) 组装用于画 ROC 的 tidy DataFrame
    roc_df = pd.DataFrame({
        "fpr":      mean_fpr,
        "tpr_mean": tpr_mean,
        "tpr_ci95": ci95,
    })

    # 附加元信息
    roc_df["model"]          = "svc_random100_cv"
    roc_df["n_runs"]         = N_RUNS
    roc_df["n_features_run"] = N_FEATURES_PER_RUN
    roc_df["test_size"]      = TEST_SIZE
    roc_df["cv_folds"]       = 5
    roc_df["param_scoring"]  = "accuracy"
    roc_df["mean_auc"]       = avg_auc

    return run_metrics, roc_df


