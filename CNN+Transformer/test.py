import torch
from torch.utils.data import DataLoader
from models.cnn_transformer import CNNTransformerClassifier
from dataset import DeepfakeDataset
from utils import compute_metrics, load_checkpoint
import config


def test():
    test_set = DeepfakeDataset(config.TEST_DIR)
    test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    model = CNNTransformerClassifier()
    load_checkpoint(model, config.CHECKPOINT_PATH)
    model = model.to(config.DEVICE)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    acc, ap = compute_metrics(all_preds, all_labels)
    print(f"[TEST] acc: {acc:.4f}, ap: {ap:.4f}")


if __name__ == '__main__':
    test()