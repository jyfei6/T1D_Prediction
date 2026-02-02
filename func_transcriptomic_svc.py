import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import ElasticNetCV
import pandas as pd


def model(dataset_anova, metadata, model_selection, run_num, dropping,
          sampling=None):
    run_num = run_num  # Number of iterations
    model_type = model_selection

    '''
    Divide Test data from nan label in the metadata
    And store into a matrix of unknown sample name
    Drop all sample in the dataset altogether
    '''

    if dropping is not None:
        batch4_mask = metadata["Batch"] == 4
        batch4_metadata = metadata.loc[batch4_mask].copy()

        batch4_names_all = batch4_metadata["Samples"].tolist()
        batch4_names_in_data = [c for c in batch4_names_all if c in dataset_anova.columns]

        batch4_data = dataset_anova.loc[:, batch4_names_in_data]
        batch4_indices = batch4_metadata.index
        batch4_sample_name = batch4_metadata["Samples"]


    if dropping == "drop_all_4":
        metadata = metadata.drop(index=batch4_indices)
        cols_to_drop = [c for c in batch4_sample_name if c in dataset_anova.columns]
        if cols_to_drop:
            dataset_anova = dataset_anova.drop(columns=cols_to_drop)
    else:
        print("Dropping parameter is not correctly assigned. Skip dropping")

    test_data = batch4_data.T
    test_data = test_data.dropna(axis=0)

    labels = LabelEncoder().fit_transform(metadata['Group'])

    y = labels
    X = dataset_anova.T

    # Model selections for the training
    if model_type == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=100)
    elif model_type == 'svc':
        # classifier = SVC(probability=True, kernel='poly', degree=2, C=1, gamma='scale', max_iter=1000)
        classifier = SVC(probability=True, kernel='linear', C=1, gamma='scale', max_iter=1000)
    elif model_type == 'logistic_regression':
        classifier = LogisticRegression(max_iter=1000, solver='lbfgs')
    elif model_type == 'elastic_net':
        # classifier = ElasticNetCV(alphas=[0.05, 0.1], l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5], cv=5)
        classifier = ElasticNet(alpha=0.05, l1_ratio=0.2)

    accuracies = []

    # Variables to store accumulated ROC curve values
    fpr_accumulated = []
    tpr_accumulated = []
    roc_auc_list = []

    # Store predictions over multiple runs
    predictions_matrix = pd.DataFrame(np.zeros((run_num, test_data.shape[0])), columns=test_data.T.columns)
    training_set_matrix = pd.DataFrame(np.zeros((run_num, X.shape[0])), columns=X.T.columns)
    sensitivities = []
    specificities = []

    for i in range(run_num):
        # In the loop, following are exceuted:
        # 1) Split of the dataset
        # 2) using SMOTE [or not]
        # 3) train the model and save the ACC for this time to the vector
        # Split dataset using stratified sampling
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None, stratify=y)
        classifier.fit(X_train, y_train)

        # Predict probabilities for ROC curve calculation
        if model_type == 'elastic_net':
            y_pred_proba = classifier.predict(X_test)
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)
            # print("Best alpha:", classifier.alpha_)
            # print("Best l1_ratio:", classifier.l1_ratio_)
        else:
            # other model
            if hasattr(classifier, "predict_proba"):
                y_pred_proba = classifier.predict_proba(X_test)[:, 1]
                y_pred = classifier.predict(X_test)
            else:
                y_pred_continuous = classifier.decision_function(X_test)
                y_pred_proba = (y_pred_continuous - y_pred_continuous.min()) / (
                        y_pred_continuous.max() - y_pred_continuous.min())
                y_pred = np.where(y_pred_proba > 0.5, 1, 0)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fpr_accumulated.append(fpr)
        tpr_accumulated.append(tpr)
        roc_auc_list.append(roc_auc_score(y_test, y_pred_proba))

        # Predict on the test set without ground truth (progressor prediction)
        predictions = classifier.predict(test_data)  # Transpose back to match input shape
        predictions_matrix.iloc[i, :] = predictions
        training_pred = classifier.predict(X)  # Transpose back to match input shape
        training_set_matrix.iloc[i, :] = training_pred

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        sensitivities.append(sensitivity)
        specificities.append(specificity)

    # All codes below is for ploting ROC curve
    # Report average accuracy
    avg_accuracy = np.mean(accuracies)
    print(f"Average Accuracy over {run_num} runs using {model_type} model: {avg_accuracy:.2f}")
    mean_fpr = np.linspace(0, 1, 500)  # 500 points for a smoother curve
    mean_tpr = np.zeros_like(mean_fpr)

    # Interpolate all TPR values to the same FPR range and calculate the mean TPR
    for fpr, tpr in zip(fpr_accumulated, tpr_accumulated):
        mean_tpr += np.interp(mean_fpr, fpr, tpr)

    mean_tpr /= run_num
    mean_auc = np.mean(roc_auc_list)
    mean_tpr[0] = 0.0
    mean_tpr[-1] = 1.0

    tpr_interp = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(fpr_accumulated, tpr_accumulated)]
    tpr_interp = np.array(tpr_interp)

    mean_tpr = np.mean(tpr_interp, axis=0)
    std_tpr = np.std(tpr_interp, axis=0)

    # calculate 95% confidence interval assuming gaussian distribution
    n = len(tpr_interp)  # run_num
    ci95 = 1.96 * std_tpr / np.sqrt(n)

    plt.figure()
    plt.plot(mean_fpr, mean_tpr, lw=2, color='black',
             label=f'Average ROC Curve (AUC = {mean_auc:.2f})')
    plt.fill_between(mean_fpr,
                     np.maximum(mean_tpr - ci95, 0),
                     np.minimum(mean_tpr + ci95, 1),
                     color='grey', alpha=0.2, label='95% CI')
    plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'Average ROC Curve for {model_type.replace("_", " ").title()} over {run_num} runs',
              fontsize=18, pad=20)
    plt.legend(loc='lower right')

    # Remove top and right frame
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)

    plt.tight_layout()
    plt.show()

    # ==========================
    # save roc curve to temp_roc folder
    # ==========================
    output_dir = "./temp_roc"
    os.makedirs(output_dir, exist_ok=True)

    roc_df = pd.DataFrame({
        "fpr": mean_fpr,
        "tpr_mean": mean_tpr,
        "tpr_ci95": ci95,
    })

    roc_df["model"] = model_type
    roc_df["run_num"] = run_num
    roc_df["dropping"] = dropping
    roc_df["mean_auc"] = mean_auc

    filename_parts = [model_type, f"runs{run_num}"]

    filename = "roc_" + "_".join(filename_parts) + ".csv"
    save_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    roc_df.to_csv(save_path, index=False)
    print(f"ROC curve saved to: {save_path}")

    progressor_counts = predictions_matrix.sum(axis=0)
    training_set_counts = training_set_matrix.sum(axis=0)
    print("Prediction finished")
    return progressor_counts, training_set_counts, accuracies, sensitivities, specificities



