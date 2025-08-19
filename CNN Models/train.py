# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import DATAset
from model import get_resnet
import os
from tqdm import tqdm
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--train_path', type=str, default='/root/Project/datasets/200kMDID/Train', help='训练数据集路径')
parser.add_argument('--val_path', type=str, default='/root/Project/datasets/200kMDID/Val', help='评估数据集路径')
parser.add_argument('--model_path', type=str, default='/root/Project/weights/CNN/200kMDID', help='训练权重存储路径')
parser.add_argument('--dataset_name', type=str, default='200kMDID', help='dataset_name')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

print("Device in use:", device)
print("Is CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 路径设置
train_path = args.train_path
val_path = args.val_path

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = DATAset(train_path, transform)
val_dataset = DATAset(val_path, transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 模型、优化器、损失函数
model = get_resnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

# 创建模型保存目录
# os.makedirs("checkpoints", exist_ok=True)

best_acc = 0.0

# 训练循环
for epoch in range(args.epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item())

    train_acc = correct / total
    logging.info(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 验证阶段
    model.eval()
    # 初始化用于评估的变量
    all_preds = []
    all_probs = []
    all_labels = []
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)
            # 收集概率、预测和标签
            probs = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()  # 预测为fake的概率
            all_probs.extend(probs)
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
    val_acc = val_correct / val_total
    # 计算额外指标
    try:
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        ap = average_precision_score(all_labels, all_probs)
    except Exception as e:
        precision = recall = f1 = auc = ap = -1  # 若发生异常则赋默认值
        print(f"[WARN] Metric calculation failed: {e}")

    logging.info(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
    print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

    # 保存模型
    # torch.save(model.state_dict(), f"checkpoints/epoch{epoch+1}_acc{val_acc:.4f}.pth")
    # torch.save(model.state_dict(), f"{args.model_path}/epoch{epoch+1}_acc{val_acc:.4f}.pth")
    if val_acc > best_acc:
        best_acc = val_acc
        # if epoch < 10:
        #     torch.save(model.state_dict(), args.model_path + "/" + args.dataset_name + "_CNN_top10.pth")
        #     print("top10 best model updated")
        # if epoch < 30:
        #     torch.save(model.state_dict(), args.model_path + "/" + args.dataset_name + "_CNN_top30.pth")
        #     print("top30 best model updated")
        # if epoch < 50:
        #     torch.save(model.state_dict(), args.model_path + "/" + args.dataset_name + "_CNN_top50.pth")
        #     print("top50 best model updated")
        torch.save(model.state_dict(), args.model_path + "/" + args.dataset_name + "_CNN_best.pth")
        print("best model updated")