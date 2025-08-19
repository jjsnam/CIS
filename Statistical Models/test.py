import argparse
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
    main()