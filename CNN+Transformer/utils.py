import torch
import random
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(preds, labels):
    acc = accuracy_score(labels, preds)
    ap = average_precision_score(labels, preds)
    return acc, ap


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)
    print(f"[INFO] Saved best model to {path}")


def load_checkpoint(model, path):
    model.load_state_dict(torch.load(path, map_location='cpu'))
    print(f"[INFO] Loaded model from {path}")