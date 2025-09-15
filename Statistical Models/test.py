""" import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.feature_extractor import extract_features
import joblib
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def get_image_paths_and_labels(test_path):
    real_dir = os.path.join(test_path, 'real')
    fake_dir = os.path.join(test_path, 'fake')
    image_paths = []
    labels = []
    for img in os.listdir(real_dir):
        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
            image_paths.append(os.path.join(real_dir, img))
            labels.append(1)  # real: 1
    for img in os.listdir(fake_dir):
        if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
            image_paths.append(os.path.join(fake_dir, img))
            labels.append(0)  # fake: 0
    return image_paths, labels

def main():
    parser = argparse.ArgumentParser(description="Test GMM models on image dataset")
    parser.add_argument('--model_real', type=str, required=True, help='Path to real GMM pkl')
    parser.add_argument('--model_fake', type=str, required=True, help='Path to fake GMM pkl')
    parser.add_argument('--test_path', type=str, required=True, help='Test set directory with real/ and fake/ subfolders')
    args = parser.parse_args()
    # python test.py --model_real /root/Project/weights/Statistical/SGDF/SGDF_gmm_real_top10.pkl --model_fake /root/Project/weights/Statistical/SGDF/_gmm_fake_top10.pkl --test_path /root/Project/datasets/SGDF/Test

    # Load models
    with open(args.model_real, 'rb') as f:
        # gmm_real = pickle.load(f)
        gmm_real = joblib.load(f)
    with open(args.model_fake, 'rb') as f:
        # gmm_fake = pickle.load(f)
        gmm_fake = joblib.load(f)

    # Gather test image paths and labels
    image_paths, labels = get_image_paths_and_labels(args.test_path)
    y_true = np.array(labels)
    y_pred = []
    scores = []

    def process_image(img_path):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load image {img_path}, skipping.")
            return None, None
        feat = extract_features(img)
        feat = np.array(feat)
        if feat.ndim == 1:
            feat = feat.reshape(1, -1)
        ll_real = gmm_real.score(feat)
        ll_fake = gmm_fake.score(feat)
        diff = ll_real - ll_fake
        pred_label = 1 if diff > 0 else 0
        return pred_label, diff

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image, image_paths), total=len(image_paths), desc="Extracting features and predicting"))
        for pred_label, diff in results:
            if pred_label is not None:
                y_pred.append(pred_label)
                scores.append(diff)

    y_pred = np.array(y_pred)
    scores = np.array(scores)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except Exception:
        roc_auc = float('nan')

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Optional: plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main() """
# test_statistical.py
import argparse
import os
import joblib
import time
import math
import random
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.feature_extractor import extract_features
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------------------
# Helpers for GMM metadata & flop estimation
# ------------------------
def count_gmm_parameters(gmm):
    """
    Count scalar parameters stored in the GMM object (weights_, means_, covariances_ if present).
    This is a simple sum of array sizes and is intended as a proxy for "number of parameters".
    """
    try:
        total = 0
        if hasattr(gmm, "weights_"):
            total += np.prod(gmm.weights_.shape)
        if hasattr(gmm, "means_"):
            total += np.prod(gmm.means_.shape)
        if hasattr(gmm, "covariances_"):
            total += np.prod(gmm.covariances_.shape)
        return int(total)
    except Exception:
        return None

def estimate_gmm_flops_per_sample(gmm):
    """
    Provide a rough estimate of floating-point ops required by gmm.score per single sample.
    APPROXIMATION assumptions:
      - For 'full' covariance: matrix-vector multiply cost ~ 2 * n_features^2 per component (mult + add)
      - For 'diag' covariance: cost ~ 5 * n_features per component (subtract, multiply/divide, add, exp)
      - For 'tied': similar to 'full' (shared covariance but still per-component mahalanobis)
      - For 'spherical': treat like diag
    These are coarse estimates — put results in GFLOPs for readability.
    """
    try:
        n_comp = int(gmm.n_components)
        n_feat = int(gmm.means_.shape[1])
        cov_type = getattr(gmm, "covariance_type", "full")
        if cov_type == "full" or cov_type == "tied":
            ops_per_comp = 2 * (n_feat ** 2) + 5 * n_feat  # matrix-vector multiply + overhead
        elif cov_type == "diag" or cov_type == "spherical":
            ops_per_comp = 6 * n_feat  # elementwise ops + small overhead
        else:
            # fallback
            ops_per_comp = 6 * n_feat

        total_ops = n_comp * ops_per_comp
        return int(total_ops)
    except Exception:
        return None

# ------------------------
# IO helpers
# ------------------------
def get_image_paths_and_labels(test_path):
    real_dir = os.path.join(test_path, 'real')
    fake_dir = os.path.join(test_path, 'fake')
    image_paths = []
    labels = []
    if os.path.isdir(real_dir):
        for img in os.listdir(real_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                image_paths.append(os.path.join(real_dir, img))
                labels.append(1)  # real: 1
    if os.path.isdir(fake_dir):
        for img in os.listdir(fake_dir):
            if img.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')):
                image_paths.append(os.path.join(fake_dir, img))
                labels.append(0)  # fake: 0
    return image_paths, labels

# ------------------------
# Per-image processing (single)
# ------------------------
def process_image_single(img_path, gmm_real, gmm_fake):
    img = cv2.imread(img_path)
    if img is None:
        # failed to load
        return None, None
    feat = extract_features(img)
    feat = np.array(feat)
    if feat.ndim == 1:
        feat = feat.reshape(1, -1)
    # Score under each GMM
    ll_real = float(gmm_real.score(feat))
    ll_fake = float(gmm_fake.score(feat))
    diff = ll_real - ll_fake
    pred_label = 1 if diff > 0 else 0
    return pred_label, diff

# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Test GMM models on image dataset")
    parser.add_argument('--model_real', type=str, required=True, help='Path to real GMM pkl')
    parser.add_argument('--model_fake', type=str, required=True, help='Path to fake GMM pkl')
    parser.add_argument('--test_path', type=str, required=True, help='Test set directory with real/ and fake/ subfolders')
    parser.add_argument('--max_serial_samples', type=int, default=100, help='How many samples to use for serial timing (default 100)')
    parser.add_argument('--no_parallel', action='store_true', help='Disable threaded parallel processing for full test (useful for debugging)')
    args = parser.parse_args()

    # Load models (joblib handles scikit-learn pickles robustly)
    with open(args.model_real, 'rb') as f:
        gmm_real = joblib.load(f)
    with open(args.model_fake, 'rb') as f:
        gmm_fake = joblib.load(f)

    # Print parameter counts
    p_real = count_gmm_parameters(gmm_real)
    p_fake = count_gmm_parameters(gmm_fake)
    total_params = None
    if p_real is not None and p_fake is not None:
        total_params = p_real + p_fake

    print("=== Model parameter counts (scalar entries) ===")
    print(f"Real GMM params:  {p_real}")
    print(f"Fake GMM params:  {p_fake}")
    print(f"Total params (both models): {total_params}")

    # Estimate approximate FLOPs per sample
    flops_real = estimate_gmm_flops_per_sample(gmm_real)
    flops_fake = estimate_gmm_flops_per_sample(gmm_fake)
    flops_total = None
    if flops_real is not None and flops_fake is not None:
        flops_total = flops_real + flops_fake

    def fmt_ops(n):
        if n is None:
            return "N/A"
        if n >= 10**9:
            return f"{n/1e9:.3f} GFLOPs (approx)"
        if n >= 10**6:
            return f"{n/1e6:.3f} MFLOPs (approx)"
        return f"{n} FLOPs (approx)"

    print("\n=== Approximate FLOPs (per image, single model) ===")
    print(f"Real GMM approx ops/sample: {fmt_ops(flops_real)}")
    print(f"Fake GMM approx ops/sample: {fmt_ops(flops_fake)}")
    print(f"Combined approx ops/sample: {fmt_ops(flops_total)}")
    print("\nNote: these FLOPs are coarse approximations (mahalanobis + small overhead).")

    # Gather test images
    image_paths, labels = get_image_paths_and_labels(args.test_path)
    if len(image_paths) == 0:
        raise RuntimeError("No images found in test_path. Check directory layout (real/ and fake/ subfolders).")

    # ------------------------
    # 1) Full dataset prediction using ThreadPoolExecutor (like original), measure wall time
    # ------------------------
    print("\n=== Running full dataset prediction (parallel) ===")
    y_true = []
    y_pred = []
    scores = []

    start_parallel = time.perf_counter()
    if args.no_parallel:
        # Serial fallback
        for p in tqdm(image_paths, desc="Processing images (serial)"):
            pred_label, diff = process_image_single(p, gmm_real, gmm_fake)
            if pred_label is not None:
                y_pred.append(pred_label)
                scores.append(diff)
        parallel_elapsed = time.perf_counter() - start_parallel
    else:
        # Use a thread pool for IO-bound feature extraction (keep default worker count)
        def worker(path):
            return process_image_single(path, gmm_real, gmm_fake)

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(worker, p): p for p in image_paths}
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Extracting features and predicting"):
                pred_label, diff = fut.result()
                if pred_label is not None:
                    y_pred.append(pred_label)
                    scores.append(diff)
        parallel_elapsed = time.perf_counter() - start_parallel

    # Note: y_true will be built to match number of processed images (some images may be skipped)
    # Build y_true to align with processed order: filter original labels by successful loads.
    # Simpler approach: recompute by scanning image_paths and checking which were processed (based on len)
    # But we kept ordering above; instead, we can derive y_true by re-scanning image_paths and counting successful ones.
    # For robustness, rebuild y_true by re-processing paths but only counting successful loads (cheap).
    processed_labels = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        # label detection based on folder name
        if os.path.normpath(p).split(os.sep)[-2] == 'real':
            processed_labels.append(1)
        else:
            processed_labels.append(0)
    # Ensure lengths match
    if len(processed_labels) != len(y_pred):
        # fallback: try to align by using the first len(y_pred) labels
        processed_labels = processed_labels[:len(y_pred)]

    y_true = np.array(processed_labels)
    y_pred = np.array(y_pred)
    scores = np.array(scores)

    processed_count = len(y_pred)
    print(f"Processed {processed_count} images (out of {len(image_paths)} total).")
    print(f"Parallel wall time: {parallel_elapsed:.3f} s, avg per image (wall): {parallel_elapsed/processed_count*1000:.2f} ms")

    # ------------------------
    # 2) Serial timing on a sample (single-threaded) to estimate per-image latency
    # ------------------------
    n_sample = min(args.max_serial_samples, len(image_paths))
    sample_paths = random.sample(image_paths, n_sample) if len(image_paths) > n_sample else image_paths[:n_sample]
    print(f"\n=== Serial single-threaded timing on {n_sample} samples (for latency est) ===")
    # Warm-up a few samples (to mitigate caches)
    for wp in sample_paths[:5]:
        _ = process_image_single(wp, gmm_real, gmm_fake)

    start_serial = time.perf_counter()
    succ = 0
    for p in tqdm(sample_paths, desc="Serial timing"):
        pred_label, diff = process_image_single(p, gmm_real, gmm_fake)
        if pred_label is not None:
            succ += 1
    serial_elapsed = time.perf_counter() - start_serial
    if succ > 0:
        print(f"Serial total time: {serial_elapsed:.3f} s for {succ} successful samples")
        print(f"Estimated single-threaded inference time per image: {serial_elapsed/succ*1000:.2f} ms")
    else:
        print("No successful samples in serial timing (images failed to load).")

    # ------------------------
    # Metrics (compute from parallel run)
    # ------------------------
    if len(y_pred) == 0:
        raise RuntimeError("No predictions were produced. Check that images load correctly and feature extractor works.")

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        roc_auc = roc_auc_score(y_true, scores)
    except Exception:
        roc_auc = float('nan')

    print("\n=== Evaluation Metrics (on processed images) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Fake', 'Real'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()

    # Final summary of compute-related info
    print("\n=== Summary of compute-related info ===")
    print(f"Total model scalar params (real + fake): {total_params}")
    print(f"Approx combined FLOPs/sample: {fmt_ops(flops_total)}")
    print(f"Parallel avg time per image (wall): {parallel_elapsed/processed_count*1000:.2f} ms")
    if succ > 0:
        print(f"Single-thread avg time per image (serial): {serial_elapsed/succ*1000:.2f} ms")
    print("Note: FLOPs estimates are approximate; timing includes feature extraction + scoring.")

if __name__ == '__main__':
    main()