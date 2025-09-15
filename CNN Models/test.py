""" # test_cnn.py
import os
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from model import get_resnet  # CNN 模型定义
from dataset import DATAset  # 直接导入你原来的 Dataset 类
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------------
# 测试函数
# ------------------------
def test_cnn(model_path, test_path, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = DATAset(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型加载
    model = get_resnet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:,1]  # 预测为 fake 的概率
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = float('nan')

    print(f"Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

# ------------------------
# 命令行入口
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="CNN 权重路径 (.pth)")
    parser.add_argument('--test_path', type=str, required=True, help="测试集路径")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    test_cnn(args.model_path, args.test_path, args.batch_size, args.device)
     """
# python test.py --model_path /root/Project/weights/CNN/SGDF/SGDF_CNN_top10.pth --test_path /root/Project/datasets/SGDF/Test

import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from ptflops import get_model_complexity_info

from model import get_resnet  # CNN 模型定义
from dataset import DATAset  # 直接导入你原来的 Dataset 类


# ------------------------
# 工具函数
# ------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------
# 测试函数
# ------------------------
def test_cnn(model_path, test_path, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    dataset = DATAset(test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型加载
    model = get_resnet(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
    y_true, y_pred, y_prob, file_paths = [], [], [], []

    with torch.no_grad():
        for i, (imgs, labels, paths) in enumerate(loader):  # 修改：Dataset 要返回 path
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 预测为 fake 的概率
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())
            file_paths.extend(paths)
    # with torch.no_grad():
    #     for imgs, labels in loader:
    #         imgs = imgs.to(device)
    #         labels = labels.to(device)
    #         outputs = model(imgs)
    #         probs = torch.softmax(outputs, dim=1)[:,1]  # 预测为 fake 的概率
    #         preds = torch.argmax(outputs, dim=1)

    #         y_true.extend(labels.cpu().numpy())
    #         y_pred.extend(preds.cpu().numpy())
    #         y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = float('nan')

    print(f"Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # ------------------------
    # 保存错误样本
    # ------------------------
    save_dir = "results/misclassified"
    os.makedirs(save_dir, exist_ok=True)

    misclassified = [(p, t, pr) for p, t, pr in zip(file_paths, y_true, y_pred) if t != pr]
    for p, t, pr in misclassified[:20]:  # 只保存前 20 张，避免太多
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
        "true_label": y_true,
        "pred_label": y_pred,
        "pred_prob": y_prob
    })
    df.to_csv(results_csv, index=False)
    print(f"Saved predictions to {results_csv}")


# ------------------------
# 命令行入口
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="CNN 权重路径 (.pth)")
    parser.add_argument('--test_path', type=str, required=True, help="测试集路径")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    test_cnn(args.model_path, args.test_path, args.batch_size, args.device)
