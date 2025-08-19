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

import argparse

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='训练的轮数')
parser.add_argument('--lr', type=float, default=0.001, help='学习率')
parser.add_argument('--train_path', type=str, default='/root/Project/datasets/200kMDID/Train', help='训练数据集路径')
parser.add_argument('--val_path', type=str, default='/root/Project/datasets/200kMDID/Val', help='评估数据集路径')
parser.add_argument('--train_cache_path', type=str, default='/root/Project/RCNN Models/cache/regions/200kMDID/Train', help='训练数据集路径')
parser.add_argument('--val_cache_path', type=str, default='/root/Project/RCNN Models/cache/regions/200kMDID/Val', help='评估数据集路径')
parser.add_argument('--model_path', type=str, default='/root/Project/weights/CNN/200kMDID', help='训练权重存储路径')
parser.add_argument('--dataset_name', type=str, default='200kMDID', help='dataset_name')

args = parser.parse_args()

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


train_dir = args.train_path
val_dir = args.val_path
train_cache_dir = args.train_cache_path
val_cache_dir = args.val_cache_path
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
optimizer = optim.Adam(model.parameters(), lr=args.lr)

EPOCHS = args.epochs
# os.makedirs("checkpoints", exist_ok=True)
best_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    all_preds = []
    all_labels = []
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
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item(), acc=correct/total)
        except IndexError as e:
            print(f"[SKIP] Skipping batch due to error: {e}")
            logging.warning(f"[SKIP] Skipping batch due to error: {e}")
            continue

    print(f"[Epoch {epoch+1}] Train Loss: {total_loss:.4f}, Train Acc: {correct/total:.4f}")
    logging.info(f"Epoch {epoch+1} | Train Loss: {total_loss:.4f}, Train Acc: {correct/total:.4f}")

    train_precision = precision_score(all_labels, all_preds)
    train_recall = recall_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds)
    train_ap = average_precision_score(all_labels, all_preds)
    train_auc = roc_auc_score(all_labels, all_preds)
    print(f"[Epoch {epoch+1}] Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, AP: {train_ap:.4f}")
    logging.info(f"Epoch {epoch+1} | Train Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}, AP: {train_ap:.4f}")
    print(f"[Epoch {epoch+1}] Train AUC: {train_auc:.4f}")
    logging.info(f"Epoch {epoch+1} | Train AUC: {train_auc:.4f}")

    # Validation
    model.eval()
    val_acc = 0.0
    with torch.no_grad():
        val_correct, val_total = 0, 0
        val_all_preds = []
        val_all_labels = []
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
                val_all_preds.extend(preds.detach().cpu().numpy())
                val_all_labels.extend(labels.detach().cpu().numpy())
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

            except Exception as e:
                print(f"[VAL SKIP] Exception during validation: {e}")
                logging.warning(f"[VAL SKIP] Exception during validation: {e}")
                continue
        print(f"[Epoch {epoch+1}] Val Acc: {val_correct/val_total:.4f}")
        logging.info(f"Epoch {epoch+1} | Val Acc: {val_correct/val_total:.4f}")
        val_acc = val_correct / val_total

        val_precision = precision_score(val_all_labels, val_all_preds)
        val_recall = recall_score(val_all_labels, val_all_preds)
        val_f1 = f1_score(val_all_labels, val_all_preds)
        val_ap = average_precision_score(val_all_labels, val_all_preds)
        val_auc = roc_auc_score(val_all_labels, val_all_preds)
        print(f"[Epoch {epoch+1}] Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AP: {val_ap:.4f}")
        logging.info(f"Epoch {epoch+1} | Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AP: {val_ap:.4f}")
        print(f"[Epoch {epoch+1}] Val AUC: {val_auc:.4f}")
        logging.info(f"Epoch {epoch+1} | Val AUC: {val_auc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        # if epoch < 10:
        #     save_path = args.model_path + "/" + args.dataset_name + "_RCNN_top10.pth"
        #     save_checkpoint(model, optimizer, epoch + 1, save_path)
        #     print(f"Saved best model top10 to {save_path}")
        #     logging.info(f"Saved best model top10 to {save_path}")
        # if epoch < 30:
        #     save_path = args.model_path + "/" + args.dataset_name + "_RCNN_top30.pth"
        #     save_checkpoint(model, optimizer, epoch + 1, save_path)
        #     print(f"Saved best model top30 to {save_path}")
        #     logging.info(f"Saved best model top30 to {save_path}")
        # if epoch < 50:
        #     save_path = args.model_path + "/" + args.dataset_name + "_RCNN_top50.pth"
        #     save_checkpoint(model, optimizer, epoch + 1, save_path)
        #     print(f"Saved best model top50 to {save_path}")
        #     logging.info(f"Saved best model top50 to {save_path}")
        save_path = args.model_path + "/" + args.dataset_name + "_RCNN_best.pth"
        save_checkpoint(model, optimizer, epoch + 1, save_path)
        print(f"Saved best model to {save_path}")
        logging.info(f"Saved best model to {save_path}")
    # save_path = os.path.join("checkpoints", f"checkpoint_rcnn_epoch{epoch+1}.pth")
    # save_path = args.model_path + "/checkpoint_rcnn_epoch{epoch+1}.pth"
    # save_checkpoint(model, optimizer, epoch + 1, save_path)
    # logging.info(f"Saved checkpoint to {save_path}")
