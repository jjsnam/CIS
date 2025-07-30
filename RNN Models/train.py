# import os
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from dataset import SequenceDataset
# from model import CNN_BiLSTM
# from tqdm import tqdm
# import logging
# import matplotlib.pyplot as plt  # 在顶部导入

# # 设置路径
# train_dir = "../dataset/Celeb_V2/Train"
# val_dir = "../dataset/Celeb_V2/Val"
# checkpoint_dir = "checkpoints"
# os.makedirs(checkpoint_dir, exist_ok=True)

# # 日志记录
# logging.basicConfig(filename="train.log", level=logging.INFO, format="%(asctime)s %(message)s")

# # 参数
# sequence_length = 5
# batch_size = 32
# epochs = 10
# lr = 1e-4
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 数据变换
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# # 数据集和数据加载器
# train_dataset = SequenceDataset(train_dir, sequence_length=sequence_length, transform=transform)
# val_dataset = SequenceDataset(val_dir, sequence_length=sequence_length, transform=transform)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
#                           num_workers=4, pin_memory=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
#                         num_workers=4, pin_memory=True)

# # 模型、损失、优化器
# model = CNN_BiLSTM().to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# # 添加用于绘图的列表
# train_acc_list = []
# val_acc_list = []

# # 训练循环
# for epoch in range(epochs):
#     model.train()
#     total_loss = 0
#     correct = 0
#     total = 0
#     loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

#     for batch in loop:
#         images, labels = batch
#         images = torch.stack(images, dim=1).to(device)  # (B, seq, C, H, W)
#         labels = labels.to(device)

#         outputs = model(images)
#         loss = criterion(outputs, labels)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()
#         preds = outputs.argmax(dim=1)
#         correct += (preds == labels).sum().item()
#         total += labels.size(0)

#         loop.set_postfix(loss=loss.item(), acc=correct/total if total else 0)

#     train_acc = correct / total
#     train_acc_list.append(train_acc)
#     logging.info(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")
#     print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

#     # 验证阶段
#     model.eval()
#     val_correct = 0
#     val_total = 0
#     with torch.no_grad():
#         for batch in val_loader:
#             images, labels = batch
#             images = torch.stack(images, dim=1).to(device)
#             labels = labels.to(device)
#             outputs = model(images)
#             preds = outputs.argmax(dim=1)
#             val_correct += (preds == labels).sum().item()
#             val_total += labels.size(0)
#     val_acc = val_correct / val_total
#     val_acc_list.append(val_acc)
#     logging.info(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")
#     print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")

#     # 保存模型
#     ckpt_path = os.path.join(checkpoint_dir, f"epoch{epoch+1}_val{val_acc:.4f}.pth")
#     torch.save(model.state_dict(), ckpt_path)

# # 训练完成后绘图
# plt.figure()
# plt.plot(range(1, epochs+1), train_acc_list, label="Train Acc")
# plt.plot(range(1, epochs+1), val_acc_list, label="Val Acc")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.title("Training and Validation Accuracy")
# plt.legend()
# plt.grid(True)
# plt.savefig("accuracy_curve.png")
# plt.close()
# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LSTMClassifier
from dataset import SequenceDataset  # 保持原有序列构造方式

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# train_dir = "/root/Project/dataset/Celeb_V2/Train"
# val_dir = "/root//Project/dataset/Celeb_V2/Val"
# train_dir = "/root/Project/dataset/OpenForensics/Train"
# val_dir = "/root//Project/dataset/OpenForensics/Val"
train_dir = "/root/Project/dataset/SGDF/Train"
val_dir = "/root//Project/dataset/SGDF/Val"
sequence_length = 5
batch_size = 8
epochs = 10

# Transforms 与之前相同
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = SequenceDataset(train_dir, sequence_length=sequence_length, transform=transform)
val_dataset = SequenceDataset(val_dir, sequence_length=sequence_length, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

model = LSTMClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

log_file = open("training.log", "w")
checkpoint_dir = "/root/Project/RNN Models/checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
    for images, labels in pbar:
        # print(images.device, labels.device)  # 确保在 cuda 上
        images = images.to(device, non_blocking=True)            # (B, T, 3, H, W)
        labels = labels.to(device, non_blocking=True)            # (B, T)

        outputs = model(images)               # (B, T, 2)
        loss = criterion(outputs.view(-1, 2), labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=2)         # (B, T)
        correct += (preds == labels).sum().item()
        total += labels.numel()
        train_loss += loss.item()

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    train_acc = correct / total
    log_file.write(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n")
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

    # === 验证 ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(images)  # (B, T, 2)
            preds = outputs.argmax(dim=2)
            correct += (preds == labels).sum().item()
            total += labels.numel()
    val_acc = correct / total
    log_file.write(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}\n")
    print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")

    # === 保存模型 ===
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth"))

log_file.close()