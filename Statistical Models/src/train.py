# src/train.py
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.feature_extractor import extract_features
from src.gmm_model import GMMClassifier
from src.utils import init_logging, get_image_paths
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from scipy.spatial.distance import jensenshannon


def load_features_from_folder(folder, cache_path=None):
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        return np.load(cache_path)

    paths = get_image_paths(folder)
    features = []
    for path in tqdm(paths, desc=f"Extracting from {folder}"):
        img = cv2.imread(path)
        if img is not None:
            feat = extract_features(img)
            features.append(feat)

    features = np.array(features)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, features)
        logger.info(f"Saved features to {cache_path}")
    return features

def split_dataset(X_real, X_fake, val_ratio=0.2):
    N_real = int(len(X_real) * (1 - val_ratio))
    N_fake = int(len(X_fake) * (1 - val_ratio))
    return (
        X_real[:N_real], X_real[N_real:], 
        X_fake[:N_fake], X_fake[N_fake:]
    )

def train_loop(train_real_dir, train_fake_dir, val_real_dir, val_fake_dir, model_dir, epochs, dataset):
    init_logging()

    # Load or extract features
    X_real_train = load_features_from_folder(train_real_dir, "features/"+ dataset + "/real_train.npy")
    X_fake_train = load_features_from_folder(train_fake_dir, "features/"+ dataset + "/fake_train.npy")
    X_real_val = load_features_from_folder(val_real_dir, "features/"+ dataset + "/real_val.npy")
    X_fake_val = load_features_from_folder(val_fake_dir, "features/"+ dataset + "/fake_val.npy")

    logger.info(f"Original feature dimension: {X_real_train.shape[1]}")

    model = GMMClassifier(n_components=3)
    best_acc = 0.0

    for epoch in range (epochs):
        logger.info(f"Epoch {epoch+1} - Training GMMs")
        model.train(X_real_train, X_fake_train)

        # Evaluate on training set
        X_train = np.vstack([X_real_train, X_fake_train])
        y_train = np.array([0] * len(X_real_train) + [1] * len(X_fake_train))
        scores_train = model.predict_proba(X_train)
        y_train_pred = (scores_train < 0).astype(int)
        train_acc = accuracy_score(y_train, y_train_pred)
        logger.info(f"Epoch {epoch+1} - Train Accuracy: {train_acc:.4f}")

        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        logger.info(f"Epoch {epoch+1} - Train Precision: {train_precision:.4f} | Recall: {train_recall:.4f}")

        train_f1 = f1_score(y_train, y_train_pred)
        train_auc = roc_auc_score(y_train, -scores_train)
        logger.info(f"Epoch {epoch+1} - Train F1: {train_f1:.4f} | AUC: {train_auc:.4f}")

        # clf = LogisticRegression(max_iter=1000000)
        # clf.fit(X_train, y_train)

        logger.info("Evaluating on validation set...")
        X_val = np.vstack([X_real_val, X_fake_val])
        y_val = np.array([0] * len(X_real_val) + [1] * len(X_fake_val))  # 0:real, 1:fake
        scores = model.predict_proba(X_val)
        y_pred = (scores < 0).astype(int)  # 负分更偏向 fake

        # 计算 JSD 分数作为新的分类依据
        # jsd_scores = []
        # for x in X_val:
        #     p_real = np.exp(model.model_real.score_samples(x.reshape(1, -1)))
        #     p_fake = np.exp(model.model_fake.score_samples(x.reshape(1, -1)))
        #     p_mixed = 0.5 * (p_real + p_fake)
        #     jsd = 0.5 * (np.log(p_real / p_mixed + 1e-10)) + 0.5 * (np.log(p_fake / p_mixed + 1e-10))
        #     jsd_scores.append(jsd.item())
        # jsd_scores = np.array(jsd_scores)
        # y_pred_jsd = (jsd_scores > np.median(jsd_scores)).astype(int)
        # jsd_acc = accuracy_score(y_val, y_pred_jsd)
        # logger.info(f"Epoch {epoch} - JSD Accuracy: {jsd_acc:.4f}")

        plt.figure()
        plt.hist(scores[y_val == 0], bins=50, alpha=0.6, label="Real")
        plt.hist(scores[y_val == 1], bins=50, alpha=0.6, label="Fake")
        plt.title(f"Epoch {epoch+1} - Validation Log-Likelihood Score Distribution")
        plt.xlabel("log(P_real) - log(P_fake)")
        plt.ylabel("Count")
        plt.legend()
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/val_score_epoch{epoch+1}.png")
        plt.close()

        acc = accuracy_score(y_val, y_pred)
        logger.info(f"Epoch {epoch+1} - Val Accuracy: {acc:.4f}")
        logger.info(f"Validation scores stats — mean: {scores.mean():.4f}, std: {scores.std():.4f}")

        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, -scores)  # scores 越小越 fake，取负号用于 AUC 正确排序
        logger.info(f"Epoch {epoch+1} - Val F1: {f1:.4f} | AUC: {auc:.4f}")

        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        logger.info(f"Epoch {epoch+1} - Val Precision: {precision:.4f} | Recall: {recall:.4f}")

        # y_val_pred_lr = clf.predict(X_val)
        # acc_lr = accuracy_score(y_val, y_val_pred_lr)
        # f1_lr = f1_score(y_val, y_val_pred_lr)
        # auc_lr = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1])

        # logger.info(f"Epoch {epoch} - Logistic Reg Val Accuracy: {acc_lr:.4f} | F1: {f1_lr:.4f} | AUC: {auc_lr:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            # logger.info(f"New best model found. Saving checkpoints...")
            # model.save(model_dir + "/gmm_real.pkl", model_dir + "/gmm_fake.pkl")
            # best_acc = acc
            best_acc = acc
            # if epoch < 10:
            #     logger.info(f"New best model found. Saving checkpoints...")
            #     model.save(model_dir + "/" + dataset + "_gmm_real_top10.pkl", model_dir + "/_gmm_fake_top10.pkl")
            # if epoch < 30:
            #     logger.info(f"New best model found. Saving checkpoints...")
            #     model.save(model_dir + "/" + dataset + "_gmm_real_top30.pkl", model_dir + "/_gmm_fake_top30.pkl")
            # if epoch < 50:
            #     logger.info(f"New best model found. Saving checkpoints...")
            #     model.save(model_dir + "/" + dataset + "_gmm_real_top50.pkl", model_dir + "/_gmm_fake_top50.pkl")
            logger.info(f"New best model found. Saving checkpoints...")
            model.save(model_dir + "/" + dataset + "_gmm_real_best.pkl", model_dir + "/_gmm_fake_best.pkl")