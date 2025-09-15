# paired_ttest.py
import argparse
import pandas as pd
from scipy.stats import ttest_rel

def main():
    parser = argparse.ArgumentParser(description="Paired t-test between CNN and CNN+Transformer results")
    parser.add_argument('--cnn_csv', type=str, required=True, help="CSV file for CNN results")
    parser.add_argument('--cnntr_csv', type=str, required=True, help="CSV file for CNN+Transformer results")
    args = parser.parse_args()

    # 读取结果
    cnn_df = pd.read_csv(args.cnn_csv)
    cnntr_df = pd.read_csv(args.cnntr_csv)

    # 按 path 对齐
    merged = pd.merge(cnn_df, cnntr_df, on="path", suffixes=("_cnn", "_cnntr"))

    # 确保标签一致
    if not all(merged["true_label_cnn"] == merged["true_label_cnntr"]):
        raise ValueError("True labels in the two CSVs are not aligned. Please check inputs.")

    # 提取预测概率
    probs_cnn = merged["pred_prob_cnn"].values
    probs_cnntr = merged["pred_prob_cnntr"].values

    # 运行配对 t 检验
    t_stat, p_val = ttest_rel(probs_cnn, probs_cnntr)

    print("=== Paired t-test (CNN vs CNN+Transformer on predicted probabilities) ===")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value:     {p_val:.4e}")

    if p_val < 0.05:
        print("=> 差异显著 (p < 0.05)")
    else:
        print("=> 差异不显著 (p >= 0.05)")

if __name__ == "__main__":
    main()
    
""" 
python paired_ttest.py  --cnn_csv /root/Project/CNN\ Models/results/predictions.csv  --cnntr_csv /root/Project/CNN+Transformer/results/predictions.csv
"""