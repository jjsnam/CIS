""" import os
import argparse
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model import get_model  # Transformer 模型定义
# from config import Config

# ------------------------
# 数据集类
# ------------------------
class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        classes = ['real', 'fake']
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_dir):
                self.samples.append(os.path.join(cls_dir, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ------------------------
# 测试函数
# ------------------------
def test_transformer(model_path, test_path, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 数据预处理，保持和训练一致
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(root=test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型加载
    model = get_model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 正类概率
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 指标计算
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = float('nan')

    print(f"Transformer Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real','fake'], yticklabels=['real','fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# ------------------------
# 命令行入口
# ------------------------
# if __name__ == "__main__":

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, required=True, help="Transformer 权重路径 (.pth)")
parser.add_argument('--test_path', type=str, required=True, help="测试集路径")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--device', type=str, default='cuda')
args = parser.parse_args()

test_transformer(args.model_path, args.test_path, args.batch_size, args.device) """
# python test.py --model_path /root/Project/weights/Transformer/SGDF/SGDF_Transformer_top10.pth --test_path /root/Project/datasets/SGDF/Test
import os
import argparse
import time
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info

from model import get_model  # Transformer 模型定义


# ------------------------
# 工具函数
# ------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------
# 测试函数
# ------------------------
def test_transformer(model_path, test_path, batch_size=32, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # 数据预处理，保持和训练一致
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(root=test_path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 模型加载
    model = get_model().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
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
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 正类概率
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # 指标计算
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    try:
        roc = roc_auc_score(y_true, y_prob)
    except:
        roc = float('nan')

    print(f"Transformer Test Results:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real','fake'], yticklabels=['real','fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# ------------------------
# 命令行入口
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Transformer 权重路径 (.pth)")
    parser.add_argument('--test_path', type=str, required=True, help="测试集路径")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    test_transformer(args.model_path, args.test_path, args.batch_size, args.device)