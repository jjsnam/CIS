
# train_test_cnn_transformer.py
""" import argparse
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

    test(args.model_path, args.test_path, args.batch_size, args.device, args.visualize_cm) """
# python test.py --model_path /root/Project/weights/CNN+Transformer/SGDF/SGDF_CNN+Transformer_top10.pth --test_path /root/Project/datasets/SGDF/Test
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ptflops import get_model_complexity_info

from dataset import DeepfakeDataset
from models.cnn_transformer import CNNTransformerClassifier
from utils import load_checkpoint


# ------------------------
# 工具函数
# ------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------
# 测试函数
# ------------------------
def test(model_path, test_path, batch_size=32, device='cuda', visualize_cm=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Dataset
    test_dataset = DeepfakeDataset(test_path)  # 确保能返回路径
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = CNNTransformerClassifier()
    load_checkpoint(model, model_path)
    model.to(device)
    model.eval()

    # ------------------------
    # 1. 模型参数量
    # ------------------------
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,}")

    # ------------------------
    # 2. FLOPs 计算
    # ------------------------
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        print(f"Computational complexity (FLOPs): {macs}")
        print(f"Number of parameters (ptflops): {params}")

    # ------------------------
    # 3. 单张推理时间
    # ------------------------
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    # 预热 GPU
    for _ in range(10):
        _ = model(dummy_input)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    end = time.time()
    avg_time = (end - start) / 100
    print(f"Inference Time per Image: {avg_time*1000:.2f} ms")

    # ------------------------
    # 正常测试
    # ------------------------
    all_labels, all_preds, all_probs, file_paths = [], [], [], []

    with torch.no_grad():
        for imgs, labels, paths in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            file_paths.extend(paths)

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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    # ------------------------
    # 保存错误样本
    # ------------------------
    save_dir = "results/misclassified"
    os.makedirs(save_dir, exist_ok=True)

    misclassified = [(p, t, pr) for p, t, pr in zip(file_paths, all_labels, all_preds) if t != pr]
    for p, t, pr in misclassified[:20]:  # 只保存前 20 张
        img = Image.open(p).convert("RGB")
        save_name = f"{os.path.basename(p)}_true{t}_pred{pr}.jpg"
        img.save(os.path.join(save_dir, save_name))
    print(f"Saved {len(misclassified[:20])} misclassified samples to {save_dir}")

    # ------------------------
    # 保存预测结果 CSV
    # ------------------------
    results_csv = "results/predictions.csv"
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    df = pd.DataFrame({
        "path": file_paths,
        "true_label": all_labels,
        "pred_label": all_preds,
        "pred_prob": all_probs
    })
    df.to_csv(results_csv, index=False)
    print(f"Saved predictions to {results_csv}")


# ------------------------
# 主入口
# ------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN+Transformer Testing')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained CNN+Transformer model checkpoint')
    parser.add_argument('--test_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_cm', action='store_true', help='Visualize confusion matrix')
    args = parser.parse_args()

    test(args.model_path, args.test_path, args.batch_size, args.device, args.visualize_cm)