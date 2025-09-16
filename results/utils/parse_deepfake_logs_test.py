#!/usr/bin/env python3
import re
import argparse
import pandas as pd
from pathlib import Path

# 正则表达式
HEADER_RE = re.compile(
    r"^Testing\s+(Fused|\w+)\s+(?:Trained\s+)?(CNN\+Transformer|CNN|RCNN|Transformer|Statistical)\s+Model\s+(?:top(\d+)|best)\s+on\s+(Fused|\w+)\s+dataset",
    re.IGNORECASE
)
ACC_RE = re.compile(r"Accuracy:\s*([\d.]+)")
PREC_RE = re.compile(r"Precision:\s*([\d.]+)")
RECALL_RE = re.compile(r"Recall:\s*([\d.]+)")
F1_RE = re.compile(r"(F1 Score|F1):\s*([\d.]+)")
AUC_RE = re.compile(r"(ROC-AUC|AUC):\s*([\d.]+)")
CONF_RE = re.compile(r"\[\[\s*(\d+)\s+(\d+)\s*\]\s*\[\s*(\d+)\s+(\d+)\s*\]\]")

def parse_test_log(log_path):
    results = []
    current_entry = None

    expect_conf = False
    pending_conf_nums = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # 识别一个新的测试块
        m = HEADER_RE.match(line)
        if m:
            if current_entry:  # 保存上一个 entry
                results.append(current_entry)
            train_ds, model, topk, test_ds = m.groups()
            topk_val = int(topk) if topk is not None else -1
            current_entry = {
                "Model": model,
                "TrainDataset": train_ds,
                "Dataset": test_ds,  # 测试数据集（保持原字段名兼容）
                "TopK": topk_val,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1": None,
                "AUC": None,
                "TN": None,
                "FP": None,
                "FN": None,
                "TP": None,
            }
            # 遇到新块时，重置混淆矩阵解析状态
            expect_conf = False
            pending_conf_nums = []
            continue

        if current_entry is None:
            continue

        if (m := ACC_RE.match(line)):
            current_entry["Accuracy"] = float(m.group(1))
        elif (m := PREC_RE.match(line)):
            current_entry["Precision"] = float(m.group(1))
        elif (m := RECALL_RE.match(line)):
            current_entry["Recall"] = float(m.group(1))
        elif (m := F1_RE.match(line)):
            current_entry["F1"] = float(m.group(2))
        elif (m := AUC_RE.match(line)):
            current_entry["AUC"] = float(m.group(2))
        elif "Confusion Matrix" in line:
            # 开启多行混淆矩阵采集模式
            expect_conf = True
            pending_conf_nums = []
            continue

        # 如在混淆矩阵采集模式，连续读取两行中的数字
        if expect_conf:
            nums = re.findall(r"\d+", line)
            if nums:
                pending_conf_nums.extend([int(n) for n in nums])
                if len(pending_conf_nums) >= 4:
                    current_entry["TN"], current_entry["FP"], current_entry["FN"], current_entry["TP"] = pending_conf_nums[:4]
                    expect_conf = False
                    pending_conf_nums = []
                    continue
            # 仍在采集混淆矩阵行，跳过其他解析
            continue

        # 兼容单行写法（如果有的话）
        elif (m := CONF_RE.search(line)):
            current_entry["TN"] = int(m.group(1))
            current_entry["FP"] = int(m.group(2))
            current_entry["FN"] = int(m.group(3))
            current_entry["TP"] = int(m.group(4))

    if current_entry:
        results.append(current_entry)

    return results


def save_results(results, outdir, outfile):
    df = pd.DataFrame(results)
    for col in ["TN", "FP", "FN", "TP"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").dropna().astype("Int64")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"{outfile}.csv"
    xlsx_path = outdir / f"{outfile}.xlsx"

    df.to_csv(csv_path, index=False)
    # df.to_excel(xlsx_path, index=False)

    print(f"Saved {len(df)} test results to:\n  {csv_path}\n  {xlsx_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse deepfake test logs into structured table")
    parser.add_argument("--log", type=str, required=True, help="Path to test log file")
    parser.add_argument("--outdir", type=str, required=True, help="Directory to save results")
    parser.add_argument("--outfile", type=str, default="test_results", help="Base filename (no extension)")
    args = parser.parse_args()

    results = parse_test_log(args.log)
    save_results(results, args.outdir, args.outfile)


if __name__ == "__main__":
    main()