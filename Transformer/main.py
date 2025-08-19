# main.py
import torch
from dataset import get_dataloaders
from config import Config
from dataset import *
from model import get_model
from train import train
from eval import evaluate
import torch.nn as nn
import torch.optim as optim
import logging
import os

def main():
    os.makedirs('logs', exist_ok=True)
    # os.makedirs('checkpoints', exist_ok=True)
    logging.basicConfig(
        filename='logs/train.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info("Training started.")

    train_loader, val_loader = get_dataloaders(Config.train_path, Config.val_path)
    model = get_model().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)

    best_acc = 0.0
    
    for epoch in range(Config.num_epochs):
        train_loss, train_acc, train_precision, train_ap, train_f1, train_auc, train_recall = train(model, train_loader, optimizer, None, criterion, epoch=epoch)
        val_loss, val_acc, val_precision, val_ap, val_f1, val_auc, val_recall = evaluate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}: Train acc {train_acc:.4f}, Val acc {val_acc:.4f}")
        print(f"Train - Precision: {train_precision:.4f}, AP: {train_ap:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Precision: {val_precision:.4f}, AP: {val_ap:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}, Recall: {val_recall:.4f}")

        logging.info(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")
        logging.info(f"Train Precision {train_precision:.4f}, AP {train_ap:.4f}, F1 {train_f1:.4f}, AUC {train_auc:.4f}, Recall {train_recall:.4f}")
        logging.info(f"Val Precision {val_precision:.4f}, AP {val_ap:.4f}, F1 {val_f1:.4f}, AUC {val_auc:.4f}, Recall {val_recall:.4f}")
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")

        if val_acc > best_acc:
            best_acc = val_acc
            # if epoch < 10:
            #     checkpoint_path = Config.model_path + "/" + Config.dataset_name + "_Transformer_top10.pth"
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print(f"Checkpoint saved: {checkpoint_path}")
            #     logging.info(f"Checkpoint saved: {checkpoint_path}")
            # if epoch < 30:
            #     checkpoint_path = Config.model_path + "/" + Config.dataset_name + "_Transformer_top30.pth"
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print(f"Checkpoint saved: {checkpoint_path}")
            #     logging.info(f"Checkpoint saved: {checkpoint_path}")
            # if epoch < 50:
            #     checkpoint_path = Config.model_path + "/" + Config.dataset_name + "_Transformer_top50.pth"
            #     torch.save(model.state_dict(), checkpoint_path)
            #     print(f"Checkpoint saved: {checkpoint_path}")
            #     logging.info(f"Checkpoint saved: {checkpoint_path}")
            checkpoint_path = Config.model_path + "/" + Config.dataset_name + "_Transformer_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            logging.info(f"Checkpoint saved: {checkpoint_path}")

    # logging.info("Training complete. Saving model.")
    # torch.save(model.state_dict(), Config.checkpoint_path)

if __name__ == '__main__':
    main()