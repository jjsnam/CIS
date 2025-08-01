# src/evaluate.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from src.feature_extractor import extract_features
from src.gmm_model import GMMClassifier
from src.utils import get_image_paths, init_logging
from loguru import logger

def evaluate(test_real_dir, test_fake_dir):
    init_logging("logs/test.log")

    model = GMMClassifier()
    model.load("checkpoints/gmm_real.pkl", "checkpoints/gmm_fake.pkl")

    real_paths = get_image_paths(test_real_dir)
    fake_paths = get_image_paths(test_fake_dir)

    X = []
    y = []

    for path in tqdm(real_paths + fake_paths, desc="Evaluating"):
        img = cv2.imread(path)
        if img is None:
            continue
        feat = extract_features(img)
        X.append(feat)
        y.append(0 if path in real_paths else 1)

    X = np.array(X)
    y = np.array(y)

    scores = model.predict_proba(X)
    y_pred = (scores < 0).astype(int)

    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, -scores)
    f1 = f1_score(y, y_pred)

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info(f"ROC AUC: {auc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")