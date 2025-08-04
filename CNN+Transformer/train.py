import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.cnn_transformer import CNNTransformerClassifier
from dataset import DeepfakeDataset
from utils import set_seed, compute_metrics, save_checkpoint
import config
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import logging


def train():
    logging.basicConfig(
        filename='train_log.txt',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    set_seed(config.SEED)

    train_set = DeepfakeDataset(config.TRAIN_DIR)
    val_set = DeepfakeDataset(config.VAL_DIR)
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    model = CNNTransformerClassifier()
    model = model.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for imgs, labels in loop:
            imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
        logging.info(f"Epoch {epoch+1} Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = -1.0
        ap = average_precision_score(all_labels, all_preds)

        print(f"(Val @ epoch {epoch}) Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
        logging.info(f"(Val @ epoch {epoch+1}) Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")

        if acc > best_acc:
            best_acc = acc
            if epoch < 10:
                save_checkpoint(model, config.CHECKPOINT_PATH + "top10.pth")
                logging.info(f"Saved checkpoint: top10.pth")
            if epoch < 30:
                save_checkpoint(model, config.CHECKPOINT_PATH + "top30.pth")
                logging.info(f"Saved checkpoint: top30.pth")
            if epoch < 50:
                save_checkpoint(model, config.CHECKPOINT_PATH + "top50.pth")
                logging.info(f"Saved checkpoint: top50.pth")


if __name__ == '__main__':
    train()