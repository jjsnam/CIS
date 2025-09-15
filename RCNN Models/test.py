""" # test_rcnn.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import RegionFaceDataset
from rcnn_model import MultiRegionRCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def test(model_path, test_root, cache_root, batch_size=32, device='cuda', visualize_cm=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Define transforms consistent with training (Resize + ToTensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset
    test_dataset = RegionFaceDataset(
        test_root,
        cache_root,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = MultiRegionRCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for regions, labels in test_loader:
            regions = {k: v.to(device) for k, v in regions.items()}
            labels = labels.to(device)

            outputs = model(regions)
            outputs = outputs.view(outputs.size(0), -1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设二分类
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RCNN model for Deepfake detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained RCNN model checkpoint')
    parser.add_argument('--test_root', type=str, required=True, help='Path to test dataset root folder (real/fake)')
    parser.add_argument('--cache_root', type=str, required=True, help='Path to pre-extracted cached region images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_cm', action='store_true', help='Visualize the confusion matrix')
    args = parser.parse_args()

    test(args.model_path, args.test_root, args.cache_root, args.batch_size, args.device, args.visualize_cm)
    # python test.py --model_path /root/Project/weights/RCNN/SGDF/SGDF_RCNN_top10.pth --test_root /root/Project/datasets/SGDF/Test --cache_root /root/Project/RCNN\ Models/cache/regions/SGDF/Test """

# test_rcnn.py
import argparse
import torch
import time
from torch.utils.data import DataLoader
from dataset import RegionFaceDataset
from rcnn_model import MultiRegionRCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from ptflops import get_model_complexity_info
import torch.nn as nn
import pandas as pd

# ------------------------
# 工具函数
# ------------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _maybe_cuda_sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()

# A small wrapper that accepts keyword args and forwards them as a single dict to your original model
class MultiRegionWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    # accept **kwargs (e.g. full=..., eyes=..., mouth=...) and pack into dict
    def forward(self, **regions):
        return self.base_model(regions)


# ------------------------
# 测试主函数
# ------------------------
def test(model_path, test_root, cache_root, batch_size=32, device='cuda', visualize_cm=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Define transforms consistent with training (Resize + ToTensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset & loader
    test_dataset = RegionFaceDataset(test_root, cache_root, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model load (main model on target device)
    model = MultiRegionRCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # ------------------------
    # Grab one representative batch from loader (for FLOPs + timing)
    # ------------------------
    try:
        iterator = iter(test_loader)
        regions_batch, labels_batch = next(iterator)  # regions_batch is a dict of tensors with shape (B, C, H, W)
    except StopIteration:
        raise RuntimeError("Test loader is empty. Ensure test dataset is not empty.")

    # Representative single-sample for FLOPs (CPU) and timing (device)
    regions_sample_cpu = {k: v[0:1].cpu() for k, v in regions_batch.items()}      # each -> (1, C, H, W) on CPU
    regions_sample_device = {k: v[0:1].to(device) for k, v in regions_batch.items()}  # on target device

    # ------------------------
    # 1. 模型参数量
    # ------------------------
    num_params = count_parameters(model)
    print(f"Model Parameters: {num_params:,}")

    # ------------------------
    # 2. FLOPs 计算（用 ptflops）
    #    - 使用 wrapper 让 ptflops 以 keyword 方式调用 wrapper(**dict)
    #    - wrapper 再把 kwargs 打包成 dict 传给你的原 model(regions)
    # ------------------------
    try:
        # create a fresh CPU copy of the model for FLOPs estimation (avoid moving the device model back/forth)
        model_cpu = MultiRegionRCNN()
        model_cpu.load_state_dict(checkpoint['model_state_dict'])
        model_cpu.to('cpu')
        model_cpu.eval()

        wrapper = MultiRegionWrapper(model_cpu)

        def input_constructor(input_res):
            # must return a dict so ptflops will call wrapper(**that_dict)
            # return the actual tensors (already shaped (1, C, H, W))
            return {k: v for k, v in regions_sample_cpu.items()}

        macs, params = get_model_complexity_info(
            wrapper,
            (3, 224, 224),  # dummy shape required by API (not used because we provided input_constructor)
            input_constructor=input_constructor,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
        print(f"Computational complexity (MACs reported by ptflops): {macs}")
        print(f"Number of parameters (ptflops): {params}")
    except Exception as e:
        print(f"[WARNING] Flops estimation was not finished successfully: {repr(e)}")
        macs, params = None, None

    # ------------------------
    # 3. 单张推理时间测量（使用真实代表性 sample，在目标 device 上）
    # ------------------------
    # warm-up
    for _ in range(10):
        _ = model(regions_sample_device)
    _maybe_cuda_sync(device)

    runs = 100
    start = time.time()
    for _ in range(runs):
        _ = model(regions_sample_device)
    _maybe_cuda_sync(device)
    end = time.time()
    avg_time = (end - start) / runs
    print(f"Inference Time per Image: {avg_time * 1000:.2f} ms")

    # ------------------------
    # 正常测试流程
    # ------------------------
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for regions, labels in test_loader:
            regions = {k: v.to(device) for k, v in regions.items()}
            labels = labels.to(device)

            outputs = model(regions)               # model expects a dict
            # 如果你的模型已经输出 (B, num_classes)，下面的 view 可能不需要
            # outputs = outputs.view(outputs.size(0), -1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设二分类，第二列为正类概率
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
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RCNN model for Deepfake detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained RCNN model checkpoint')
    parser.add_argument('--test_root', type=str, required=True, help='Path to test dataset root folder (real/fake)')
    parser.add_argument('--cache_root', type=str, required=True, help='Path to pre-extracted cached region images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_cm', action='store_true', help='Visualize the confusion matrix')
    args = parser.parse_args()

    test(args.model_path, args.test_root, args.cache_root, args.batch_size, args.device, args.visualize_cm)