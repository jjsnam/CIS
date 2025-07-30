import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm
from dataset import RegionFaceDataset
from rcnn_model import MultiRegionRCNN
from utils import save_checkpoint

import logging
from datetime import datetime

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    filename=log_path,
    filemode='w',
    format='%(asctime)s | %(levelname)s | %(message)s',
    level=logging.INFO
)


train_dir = "/root/Project/datasets/Celeb_V2/Train"
val_dir = "/root/Project/datasets/Celeb_V2/Val"
train_cache_dir = "/root/Project/RCNN Models/cache/regions/Celeb_V2/Train"
val_cache_dir = "/root/Project/RCNN Models/cache/regions/Celeb_V2/Val"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_ds = RegionFaceDataset(train_dir, train_cache_dir, transform)
val_ds = RegionFaceDataset(val_dir, val_cache_dir, transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4)

model = MultiRegionRCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

EPOCHS = 10
os.makedirs("checkpoints", exist_ok=True)
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        regions, labels = batch
        regions = {k: v.to(device) for k, v in regions.items()}
        labels = labels.to(device)

        # Check for empty or invalid regions before forward pass
        skip_batch = False
        for k, feats in regions.items():
            if feats is None or not hasattr(feats, "size"):
                skip_batch = True
                break
            if feats.numel() == 0 or len(feats.size()) < 2:
                skip_batch = True
                break
        if skip_batch:
            print("[SKIP] Skipping batch due to empty or invalid region input.")
            logging.warning("[SKIP] Skipping batch due to empty or invalid region input.")
            continue

        optimizer.zero_grad()
        try:
            outputs = model(regions)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)
        except IndexError as e:
            print(f"[SKIP] Skipping batch due to error: {e}")
            logging.warning(f"[SKIP] Skipping batch due to error: {e}")
            continue

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {correct/total:.4f}")
    logging.info(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f}, Train Acc: {correct/total:.4f}")

    # Validation
    model.eval()
    with torch.no_grad():
        val_correct, val_total = 0, 0
        for batch in val_loader:
            try:
                regions, labels = batch

                # 判断是否为空或格式非法
                skip_batch = False
                for k, feats in regions.items():
                    if feats is None or not hasattr(feats, "size"):
                        skip_batch = True
                        break
                    if feats.numel() == 0 or len(feats.size()) < 2:
                        skip_batch = True
                        break
                if skip_batch:
                    print("[VAL SKIP] Skipping batch due to invalid region input.")
                    logging.warning("[VAL SKIP] Skipping batch due to invalid region input.")
                    continue

                regions = {k: v.to(device) for k, v in regions.items()}
                labels = labels.to(device)
                outputs = model(regions)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

            except Exception as e:
                print(f"[VAL SKIP] Exception during validation: {e}")
                logging.warning(f"[VAL SKIP] Exception during validation: {e}")
                continue
        print(f"[Epoch {epoch+1}] Val Acc: {val_correct/val_total:.4f}")
        logging.info(f"Epoch {epoch+1} | Val Acc: {val_correct/val_total:.4f}")

    save_path = os.path.join("checkpoints", f"checkpoint_rcnn_epoch{epoch+1}.pth")
    save_checkpoint(model, optimizer, epoch + 1, save_path)
    logging.info(f"Saved checkpoint to {save_path}")
