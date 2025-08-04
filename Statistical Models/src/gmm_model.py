# src/gmm_model.py
import joblib
import numpy as np
from sklearn.mixture import GaussianMixture
from loguru import logger
import os


class GMMClassifier:
    def __init__(self, n_components=5):
        self.model_real = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=300)
        self.model_fake = GaussianMixture(n_components=n_components, covariance_type='full', max_iter=300)

    def train(self, X_real, X_fake):
        logger.info("Training GMM on real features...")
        self.model_real.fit(X_real)
        logger.info("Training GMM on fake features...")
        self.model_fake.fit(X_fake)

    def save(self, path_real, path_fake):
        joblib.dump(self.model_real, path_real)
        joblib.dump(self.model_fake, path_fake)
        logger.info(f"Saved models to {path_real}, {path_fake}")

    def load(self, path_real, path_fake):
        self.model_real = joblib.load(path_real)
        self.model_fake = joblib.load(path_fake)
        logger.info(f"Loaded models from {path_real}, {path_fake}")

    def predict_proba(self, X):
        """
        返回 real/fake 的 log likelihood 差值
        """
        log_real = self.model_real.score_samples(X)
        log_fake = self.model_fake.score_samples(X)
        return log_real - log_fake