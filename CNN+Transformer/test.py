# import torch
# from torch.utils.data import DataLoader
# from models.cnn_transformer import CNNTransformerClassifier
# from dataset import DeepfakeDataset
# from utils import compute_metrics, load_checkpoint
# import config


# def test():
#     test_set = DeepfakeDataset(config.TEST_DIR)
#     test_loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

#     model = CNNTransformerClassifier()
#     load_checkpoint(model, config.CHECKPOINT_PATH)
#     model = model.to(config.DEVICE)
#     model.eval()

#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for imgs, labels in test_loader:
#             imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
#             outputs = model(imgs)
#             preds = outputs.argmax(dim=1)
#             all_preds.extend(preds.cpu().tolist())
#             all_labels.extend(labels.cpu().tolist())

#     acc, ap = compute_metrics(all_preds, all_labels)
#     print(f"[TEST] acc: {acc:.4f}, ap: {ap:.4f}")


# if __name__ == '__main__':
#     test()
# train_test_cnn_transformer.py
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from dataset import DeepfakeDataset
from models.cnn_transformer import CNNTransformerClassifier
from utils import load_checkpoint

# ------------------------
# 测试函数
# ------------------------
def test(model_path, test_path, batch_size=32, device='cuda', visualize_cm=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Dataset
    test_dataset = DeepfakeDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = CNNTransformerClassifier()
    load_checkpoint(model, model_path)
    model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    print(f"[TEST] Accuracy:  {acc:.4f}")
    print(f"[TEST] Precision: {precision:.4f}")
    print(f"[TEST] Recall:    {recall:.4f}")
    print(f"[TEST] F1 Score:  {f1:.4f}")
    print(f"[TEST] ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("[TEST] Confusion Matrix:")
    print(cm)

    if visualize_cm:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

# ------------------------
# 主入口
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN+Transformer Training and Testing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained CNN+Transformer model checkpoint')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_cm', action='store_true', help='Visualize confusion matrix')
    args = parser.parse_args()

    test(args.model_path, args.test_path, args.batch_size, args.device, args.visualize_cm)
# python test.py --model_path /root/Project/weights/CNN+Transformer/SGDF/SGDF_CNN+Transformer_top10.pth --test_path /root/Project/datasets/SGDF/Test