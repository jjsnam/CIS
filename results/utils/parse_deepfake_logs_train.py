#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parse deepfake detection training logs of multiple model types into a wide table:
- Rows: (Model, Dataset, Metric)
- Columns: Epoch1, Epoch2, ..., EpochN
- Values: metric values per epoch
Required Val metrics: Acc, Precision, Recall, F1, AUC, AP
Optional Train metrics: Acc, Precision, Loss  (other train metrics are ignored by design)
Datasets per model:
- If 1 run: Fused
- If 2-3 runs: mapped in order to [SGDF, OpenForensics, 200kMDID]
- If >3 runs: the first 3 as above; extras named Extra_4, Extra_5, ...
Special format support:
- CNN standard "[Epoch k] ..."
- RCNN split "Train Precision", "Train AUC", "Val Acc", "Val Precision", "Val AUC"
- Transformer with "Epoch k: Train acc ..., Val acc ..." + "Train - Precision: ..." + "Val - Precision: ..."
- CNN+Transformer "(Val @ epoch 0) Acc: ..., Prec: ..., Rec: ..., F1: ..., AUC: ..., AP: ..." (epoch starts at 0) -> shift +1
Outputs: results_wide.csv and results_wide.xlsx
Usage:
    python parse_deepfake_logs.py --log /path/to/train.log --outdir /path/to/outdir
"""
import argparse
import os
import re
import sys
from collections import defaultdict, OrderedDict
import math

try:
    import pandas as pd
except Exception as e:
    print("This script requires pandas. Install via: pip install pandas openpyxl", file=sys.stderr)
    raise

MODEL_HEADER_RE = re.compile(r'^Starting training\s+(.+?)\s+model\s*$')
START_TIME_RE = re.compile(r'^Start time:\s+')
END_TIME_RE = re.compile(r'^End time:\s+')
FINISH_MODEL_RE = re.compile(r'^Finish training\s+(.+?)\s+model\s*$')

# Statistical model patterns
STAT_TRAIN_ACC_RE = re.compile(r'^- Epoch\s+(\d+)\s+-\s+Train\s+Accuracy:\s*([\d.]+)\s*$')
STAT_TRAIN_PREC_RE = re.compile(r'^- Epoch\s+(\d+)\s+-\s+Train\s+Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*$')
STAT_TRAIN_F1_AUC_RE = re.compile(r'^- Epoch\s+(\d+)\s+-\s+Train\s+F1:\s*([\d.]+)\s*\|\s*AUC:\s*([\d.]+)\s*$')
STAT_VAL_ACC_RE = re.compile(r'^- Epoch\s+(\d+)\s+-\s+Val\s+Accuracy:\s*([\d.]+)\s*$')
STAT_VAL_F1_AUC_RE = re.compile(r'^- Epoch\s+(\d+)\s+-\s+Val\s+F1:\s*([\d.]+)\s*\|\s*AUC:\s*([\d.]+)\s*$')
STAT_VAL_PREC_RECALL_RE = re.compile(r'^- Epoch\s+(\d+)\s+-\s+Val\s+Precision:\s*([\d.]+)\s*\|\s*Recall:\s*([\d.]+)\s*$')

# CNN patterns
CNN_TRAIN_RE = re.compile(
    r'^\[Epoch\s+(\d+)\]\s+Train\s+Loss:\s*([\d.]+),\s*Train\s+Acc:\s*([\d.]+)\s*$'
)
CNN_VAL_RE = re.compile(
    r'^\[Epoch\s+(\d+)\]\s+Val\s+Acc:\s*([\d.]+),\s*Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AUC:\s*([\d.]+),\s*AP:\s*([\d.]+)\s*$'
)

# RCNN patterns
RCNN_TRAIN_BASE_RE = re.compile(r'^\[Epoch\s+(\d+)\]\s+Train\s+Loss:\s*([\d.]+),\s*Train\s+Acc:\s*([\d.]+)\s*$')
RCNN_TRAIN_PREC_LINE_RE = re.compile(
    r'^\[Epoch\s+(\d+)\]\s+Train\s+Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AP:\s*([\d.]+)\s*$'
)
RCNN_TRAIN_AUC_LINE_RE = re.compile(r'^\[Epoch\s+(\d+)\]\s+Train\s+AUC:\s*([\d.]+)\s*$')
RCNN_VAL_ACC_RE = re.compile(r'^\[Epoch\s+(\d+)\]\s+Val\s+Acc:\s*([\d.]+)\s*$')
RCNN_VAL_PREC_LINE_RE = re.compile(
    r'^\[Epoch\s+(\d+)\]\s+Val\s+Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AP:\s*([\d.]+)\s*$'
)
RCNN_VAL_AUC_LINE_RE = re.compile(r'^\[Epoch\s+(\d+)\]\s+Val\s+AUC:\s*([\d.]+)\s*$')

# Transformer patterns
TRF_EPOCH_LINE_RE = re.compile(
    r'^Epoch\s+(\d+):\s*Train\s+acc\s+([\d.]+),\s*Val\s+acc\s+([\d.]+)\s*$'
)
TRF_TRAIN_DETAIL_RE = re.compile(
    r'^\s*Train\s*-\s*Precision:\s*([\d.]+),\s*AP:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AUC:\s*([\d.]+),\s*Recall:\s*([\d.]+)\s*$'
)
TRF_VAL_DETAIL_RE = re.compile(
    r'^\s*Val\s*-\s*Precision:\s*([\d.]+),\s*AP:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AUC:\s*([\d.]+),\s*Recall:\s*([\d.]+)\s*$'
)

# CNN+Transformer (Val-only)
CNNT_VAL_RE = re.compile(
    r'^\(Val\s*@\s*epoch\s*(\d+)\)\s*Acc:\s*([\d.]+),\s*Prec:\s*([\d.]+),\s*Rec:\s*([\d.]+),\s*F1:\s*([\d.]+),\s*AUC:\s*([\d.]+),\s*AP:\s*([\d.]+)\s*$'
)

DATASET_ORDER = ["SGDF", "OpenForensics", "200kMDID"]

def normalize_model_name(name: str) -> str:
    name = name.strip()
    # Normalize spaces and plus sign variations
    name = name.replace("＋", "+").replace(" ", "")
    # Recover expected display names
    if name.lower() in {"cnn"}:
        return "CNN"
    if name.lower() in {"rcnn"}:
        return "RCNN"
    if name.lower() in {"transformer"}:
        return "Transformer"
    if name.lower() in {"cnn+transformer", "cnntransformer", "cnn-transformer"}:
        return "CNN+Transformer"
    if name.lower() in {"statistical", "stat"}:
        return "Statistical"
    return name

def dataset_name_for_index(idx: int, total_runs: int) -> str:
    if total_runs == 1:
        return "Fused"
    if idx < len(DATASET_ORDER):
        return DATASET_ORDER[idx]
    return f"Extra_{idx+1}"

def make_key(model, dataset, split, metric):
    # split ∈ {"Val","Train"}
    # metric canonical names
    return (model, dataset, f"{split} {metric}")

def set_value(store, model, dataset, split, metric, epoch, value):
    key = make_key(model, dataset, split, metric)
    store[key][epoch] = value

def parse_log(lines):
    """
    Returns:
        store: dict[(Model, Dataset, "Split Metric")] -> dict[epoch(int)->value(float)]
    """
    store = defaultdict(lambda: defaultdict(lambda: math.nan))
    current_model = None
    current_model_runs = []  # list of run indices encountered (we count "Start time" blocks)
    run_index = -1
    # To defer dataset naming until we know count per model, we keep per-run collected epochs,
    # and after finishing a model we map run_id -> dataset name.
    per_model_per_run_records = defaultdict(list)  # model -> list of (run_id, (setter function))
    # Instead of setter function, collect entries temporary with run_id
    temp_entries = []  # list of dicts: {model, run_id, split, metric, epoch, value}

    # Transformer epoch context
    trf_pending_epoch = None
    trf_in_model = False

    # For CNN+Transformer epoch shift
    cnnt_epoch_shift_needed = False

    # Track per model whether we're inside a model block (from "Starting..." to "Finish..." or next "Starting...")
    inside_model_block = False

    for raw_line in lines:
        line = raw_line.strip()

        m = MODEL_HEADER_RE.match(line)
        if m:
            # Starting a new model
            if inside_model_block:
                # reset per-model context when new model starts without explicit finish
                trf_pending_epoch = None
            inside_model_block = True
            current_model = normalize_model_name(m.group(1))
            run_index = -1
            current_model_runs = []
            trf_in_model = (current_model == "Transformer")
            cnnt_epoch_shift_needed = (current_model == "CNN+Transformer")
            continue

        if current_model is None:
            # not yet inside any model
            continue

        if START_TIME_RE.match(line):
            run_index += 1
            current_model_runs.append(run_index)
            # Reset epoch context for Transformer within each run
            trf_pending_epoch = None
            continue

        if FINISH_MODEL_RE.match(line) or MODEL_HEADER_RE.match(line):
            # We will remap run_id to dataset names and flush temp entries; but do it after loop too.
            pass

        # ------- Parse by model -------
        if current_model == "CNN":
            m1 = CNN_TRAIN_RE.match(line)
            if m1:
                epoch = int(m1.group(1))
                loss = float(m1.group(2))
                acc = float(m1.group(3))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Loss", epoch=epoch, value=loss))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Acc", epoch=epoch, value=acc))
                continue
            m2 = CNN_VAL_RE.match(line)
            if m2:
                epoch = int(m2.group(1))
                acc = float(m2.group(2)); prec=float(m2.group(3)); rec=float(m2.group(4))
                f1=float(m2.group(5)); auc=float(m2.group(6)); ap=float(m2.group(7))
                temp_entries.extend([
                    dict(model=current_model, run_id=run_index, split="Val", metric="Acc", epoch=epoch, value=acc),
                    dict(model=current_model, run_id=run_index, split="Val", metric="Precision", epoch=epoch, value=prec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="Recall", epoch=epoch, value=rec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="F1", epoch=epoch, value=f1),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AUC", epoch=epoch, value=auc),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AP", epoch=epoch, value=ap),
                ])
                continue

        elif current_model == "RCNN":
            m1 = RCNN_TRAIN_BASE_RE.match(line)
            if m1:
                epoch = int(m1.group(1)); loss=float(m1.group(2)); acc=float(m1.group(3))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Loss", epoch=epoch, value=loss))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Acc", epoch=epoch, value=acc))
                continue
            m2 = RCNN_TRAIN_PREC_LINE_RE.match(line)
            if m2:
                epoch = int(m2.group(1)); prec=float(m2.group(2))
                # Optional: only keep Train Precision (ignore Train Recall/F1/AP per spec)
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Precision", epoch=epoch, value=prec))
                continue
            m3 = RCNN_VAL_ACC_RE.match(line)
            if m3:
                epoch = int(m3.group(1)); acc=float(m3.group(2))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="Acc", epoch=epoch, value=acc))
                continue
            m4 = RCNN_VAL_PREC_LINE_RE.match(line)
            if m4:
                epoch = int(m4.group(1)); prec=float(m4.group(2)); rec=float(m4.group(3)); f1=float(m4.group(4)); ap=float(m4.group(5))
                temp_entries.extend([
                    dict(model=current_model, run_id=run_index, split="Val", metric="Precision", epoch=epoch, value=prec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="Recall", epoch=epoch, value=rec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="F1", epoch=epoch, value=f1),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AP", epoch=epoch, value=ap),
                ])
                continue
            m5 = RCNN_VAL_AUC_LINE_RE.match(line)
            if m5:
                epoch = int(m5.group(1)); auc=float(m5.group(2))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="AUC", epoch=epoch, value=auc))
                continue

        elif current_model == "Transformer":
            m1 = TRF_EPOCH_LINE_RE.match(line)
            if m1:
                trf_pending_epoch = int(m1.group(1))
                train_acc = float(m1.group(2)); val_acc = float(m1.group(3))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Acc", epoch=trf_pending_epoch, value=train_acc))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="Acc", epoch=trf_pending_epoch, value=val_acc))
                continue
            m2 = TRF_TRAIN_DETAIL_RE.match(line)
            if m2 and trf_pending_epoch is not None:
                prec=float(m2.group(1))
                # other fields exist but we only keep Train Precision per spec
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Precision", epoch=trf_pending_epoch, value=prec))
                continue
            m3 = TRF_VAL_DETAIL_RE.match(line)
            if m3 and trf_pending_epoch is not None:
                prec=float(m3.group(1)); ap=float(m3.group(2)); f1=float(m3.group(3)); auc=float(m3.group(4)); rec=float(m3.group(5))
                temp_entries.extend([
                    dict(model=current_model, run_id=run_index, split="Val", metric="Precision", epoch=trf_pending_epoch, value=prec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="Recall", epoch=trf_pending_epoch, value=rec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="F1", epoch=trf_pending_epoch, value=f1),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AUC", epoch=trf_pending_epoch, value=auc),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AP", epoch=trf_pending_epoch, value=ap),
                ])
                continue

        elif current_model == "CNN+Transformer":
            m1 = CNNT_VAL_RE.match(line)
            if m1:
                epoch = int(m1.group(1)) + 1  # Shift to start from 1
                acc = float(m1.group(2)); prec=float(m1.group(3)); rec=float(m1.group(4))
                f1=float(m1.group(5)); auc=float(m1.group(6)); ap=float(m1.group(7))
                temp_entries.extend([
                    dict(model=current_model, run_id=run_index, split="Val", metric="Acc", epoch=epoch, value=acc),
                    dict(model=current_model, run_id=run_index, split="Val", metric="Precision", epoch=epoch, value=prec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="Recall", epoch=epoch, value=rec),
                    dict(model=current_model, run_id=run_index, split="Val", metric="F1", epoch=epoch, value=f1),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AUC", epoch=epoch, value=auc),
                    dict(model=current_model, run_id=run_index, split="Val", metric="AP", epoch=epoch, value=ap),
                ])
                continue

        elif current_model == "Statistical":
            m1 = STAT_TRAIN_ACC_RE.match(line)
            if m1:
                epoch = int(m1.group(1)); acc=float(m1.group(2))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Acc", epoch=epoch, value=acc))
                continue
            m2 = STAT_TRAIN_PREC_RE.match(line)
            if m2:
                epoch = int(m2.group(1)); prec=float(m2.group(2))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Train", metric="Precision", epoch=epoch, value=prec))
                continue
            m3 = STAT_TRAIN_F1_AUC_RE.match(line)
            if m3:
                epoch = int(m3.group(1))
                # skip Train F1 and AUC because not in optional metrics
                continue
            m4 = STAT_VAL_ACC_RE.match(line)
            if m4:
                epoch = int(m4.group(1)); acc=float(m4.group(2))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="Acc", epoch=epoch, value=acc))
                continue
            m5 = STAT_VAL_F1_AUC_RE.match(line)
            if m5:
                epoch = int(m5.group(1)); f1=float(m5.group(2)); auc=float(m5.group(3))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="F1", epoch=epoch, value=f1))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="AUC", epoch=epoch, value=auc))
                continue
            m6 = STAT_VAL_PREC_RECALL_RE.match(line)
            if m6:
                epoch = int(m6.group(1)); prec=float(m6.group(2)); rec=float(m6.group(3))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="Precision", epoch=epoch, value=prec))
                temp_entries.append(dict(model=current_model, run_id=run_index, split="Val", metric="Recall", epoch=epoch, value=rec))
                continue
        # otherwise ignore the line

    # ---- Assign dataset names per model and populate store ----
    # Count runs per model
    model_to_run_count = defaultdict(int)
    for e in temp_entries:
        model_to_run_count[e['model']] = max(model_to_run_count[e['model']], e['run_id'] + 1)

    # Now map run_id->dataset name
    for e in temp_entries:
        model = e['model']
        total_runs = model_to_run_count[model] if model in model_to_run_count and model_to_run_count[model] > 0 else 1
        dataset = dataset_name_for_index(e['run_id'] if e['run_id'] is not None and e['run_id']>=0 else 0, total_runs)
        split = e['split']
        metric = e['metric']
        epoch = int(e['epoch'])
        value = float(e['value'])
        set_value(store, model, dataset, split, metric, epoch, value)

    return store

def build_wide_dataframe(store):
    """
    Input store: dict[(Model, Dataset, "Split Metric")] -> dict[epoch->value]
    Output: pandas DataFrame with MultiIndex (Model, Dataset, Metric) and columns Epoch1..EpochN (global max epoch)
    We keep only the required Val metrics (Acc, Precision, Recall, F1, AUC, AP) and the optional Train metrics (Acc, Precision, Loss).
    """
    # Collect all epochs to determine max
    max_epoch = 0
    for _, epoch_map in store.items():
        if epoch_map:
            max_epoch = max(max_epoch, max(epoch_map.keys()))
    if max_epoch == 0:
        max_epoch = 1

    # Prepare rows
    rows = []
    idx = []
    desired_metrics = (
        [("Val", "Acc"), ("Val", "Precision"), ("Val", "Recall"), ("Val", "F1"), ("Val", "AUC"), ("Val", "AP")] +
        [("Train", "Acc"), ("Train", "Precision"), ("Train", "Loss")]
    )

    # Order rows deterministically
    sorted_keys = sorted(store.keys(), key=lambda k: (k[0], k[1], k[2]))
    # Build a lookup for quick access
    key_to_epochmap = {k: v for k, v in store.items()}

    # To ensure that even missing (Model, Dataset, Metric) rows appear, we first discover all present (Model,Dataset)
    present_pairs = set((k[0], k[1]) for k in sorted_keys)
    for model, dataset in sorted(present_pairs):
        for split, metric in desired_metrics:
            label = f"{split} {metric}"
            k = (model, dataset, label)
            epoch_map = key_to_epochmap.get(k, {})
            row = []
            for ep in range(1, max_epoch + 1):
                row.append(epoch_map.get(ep, float('nan')))
            rows.append(row)
            idx.append((model, dataset, label))

    columns = [f"Epoch{ep}" for ep in range(1, max_epoch + 1)]
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(idx, names=["Model", "Dataset", "Metric"]), columns=columns)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, help="Path to the training log file")
    parser.add_argument("--outdir", default=".", help="Directory to save output files")
    parser.add_argument("--outfile", default="results_wide", help="Base filename (without extension) for output files")
    args = parser.parse_args()

    log_path = args.log
    outdir = args.outdir
    outfile_base = args.outfile

    if not os.path.isfile(log_path):
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    store = parse_log(lines)
    df = build_wide_dataframe(store)

    csv_path = os.path.join(outdir, f"{outfile_base}.csv")
    xlsx_path = os.path.join(outdir, f"{outfile_base}.xlsx")
    df.to_csv(csv_path, encoding="utf-8")
    # For Excel, ensure engine available
    try:
        df.to_excel(xlsx_path, merge_cells=False)
    except Exception as e:
        print("Failed to write .xlsx (try `pip install openpyxl`). Writing CSV only.", file=sys.stderr)

    print(f"Saved: {csv_path}")
    if os.path.exists(xlsx_path):
        print(f"Saved: {xlsx_path}")

if __name__ == "__main__":
    main()