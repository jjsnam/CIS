# train.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CelebV2Dataset
from model import get_resnet
import os
from tqdm import tqdm
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(filename='train.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

print("Device in use:", device)
print("Is CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 路径设置
train_path = "/root/Project/datasets/Celeb_V2/Train"
val_path = "/root/Project/datasets/Celeb_V2/Val"

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = CelebV2Dataset(train_path, transform)
val_dataset = CelebV2Dataset(val_path, transform)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

# 模型、优化器、损失函数
model = get_resnet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# 创建模型保存目录
os.makedirs("checkpoints", exist_ok=True)

# 训练循环
for epoch in range(10):
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
    val_acc = val_correct / val_total
    logging.info(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")
    print(f"[Epoch {epoch+1}] Val Acc: {val_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), f"checkpoints/epoch{epoch+1}_acc{val_acc:.4f}.pth")