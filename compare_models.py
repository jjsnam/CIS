import argparse
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.metrics import roc_auc_score

# --- DeLong’s test implementation (public domain) ---
from scipy import stats

def delong_roc_variance(ground_truth, predictions):
    # ground_truth: 0/1, predictions: prob
    order = np.argsort(-predictions)
    predictions = predictions[order]
    ground_truth = ground_truth[order]

    distinct_value_indices = np.where(np.diff(predictions))[0]
    threshold_idxs = np.r_[distinct_value_indices, ground_truth.size - 1]

    tpr = np.cumsum(ground_truth)[threshold_idxs] / ground_truth.sum()
    fpr = (1 + threshold_idxs - np.cumsum(ground_truth)[threshold_idxs]) / (len(ground_truth) - ground_truth.sum())

    V = np.trapz(tpr, fpr)
    return V

def delong_test(y_true, pred1, pred2):
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    var1 = delong_roc_variance(y_true, pred1)
    var2 = delong_roc_variance(y_true, pred2)
    se = np.sqrt(var1 + var2)
    z = (auc1 - auc2) / se
    p = 2 * stats.norm.sf(abs(z))
    return auc1, auc2, z, p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn_csv', type=str, required=True)
    parser.add_argument('--cnntr_csv', type=str, required=True)
    args = parser.parse_args()

    cnn = pd.read_csv(args.cnn_csv)
    cnntr = pd.read_csv(args.cnntr_csv)

    # 对齐 path
    merged = pd.merge(cnn, cnntr, on="path", suffixes=("_cnn", "_cnntr"))

    y_true = merged["true_label_cnn"].values
    pred_cnn = merged["pred_label_cnn"].values
    pred_cnntr = merged["pred_label_cnntr"].values
    prob_cnn = merged["pred_prob_cnn"].values
    prob_cnntr = merged["pred_prob_cnntr"].values

    # --- McNemar’s test ---
    b = np.sum((pred_cnn == y_true) & (pred_cnntr != y_true))  # CNN correct, CNNtr wrong
    c = np.sum((pred_cnn != y_true) & (pred_cnntr == y_true))  # CNN wrong, CNNtr correct
    table = [[0, b], [c, 0]]
    result = mcnemar(table, exact=False, correction=True)

    print("=== McNemar’s Test ===")
    print(f"b = {b}, c = {c}")
    print(f"statistic = {result.statistic:.4f}, p-value = {result.pvalue:.4e}")
    if result.pvalue < 0.05:
        print("=> 差异显著 (p < 0.05)")
    else:
        print("=> 差异不显著 (p >= 0.05)")

    # --- DeLong’s test for AUC ---
    auc1, auc2, z, p = delong_test(y_true, prob_cnn, prob_cnntr)
    print("\n=== DeLong’s Test for AUC ===")
    print(f"AUC CNN:           {auc1:.4f}")
    print(f"AUC CNN+Transformer: {auc2:.4f}")
    print(f"z = {z:.4f}, p-value = {p:.4e}")

if __name__ == "__main__":
    main()