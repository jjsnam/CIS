import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import SequenceDataset
from model import CNN_BiLSTM
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt  # 在顶部导入

# 设置路径
train_dir = "../datasets/Celeb_V2/IdentitySplit/Train"
val_dir = "../datasets/Celeb_V2/IdentitySplit/Val"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 日志记录
logging.basicConfig(filename="train.log", level=logging.INFO, format="%(asctime)s %(message)s")

# 参数
sequence_length = 5
batch_size = 32
epochs = 10
lr = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 数据集和数据加载器
train_dataset = SequenceDataset(train_dir, sequence_length=sequence_length, transform=transform)
val_dataset = SequenceDataset(val_dir, sequence_length=sequence_length, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

# 模型、损失、优化器
model = CNN_BiLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 添加用于绘图的列表
train_acc_list = []
val_acc_list = []

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in loop:
        images, labels = batch
        images = torch.stack(images, dim=1).to(device)  # (B, seq, C, H, W)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=correct/total if total else 0)

    train_acc = correct / total
    train_acc_list.append(train_acc)
    logging.info(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}")

    # 验证阶段
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_loader:
            images, labels = batch
            images = torch.stack(images, dim=1).to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
    val_acc = val_correct / val_total
    val_acc_list.append(val_acc)
    logging.info(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")
    print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")

    # 保存模型
    ckpt_path = os.path.join(checkpoint_dir, f"epoch{epoch+1}_val{val_acc:.4f}.pth")
    torch.save(model.state_dict(), ckpt_path)

# 训练完成后绘图
plt.figure()
plt.plot(range(1, epochs+1), train_acc_list, label="Train Acc")
plt.plot(range(1, epochs+1), val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("accuracy_curve.png")
plt.close()