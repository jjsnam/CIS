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
    os.makedirs('checkpoints', exist_ok=True)
    logging.basicConfig(
        filename='logs/train.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    logging.info("Training started.")

    train_loader, val_loader = get_dataloaders()
    model = get_model().to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)

    for epoch in range(Config.num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, None, criterion, epoch=epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}: Train acc {train_acc:.4f}, Val acc {val_acc:.4f}")
        logging.info(f"Epoch {epoch+1}: Train loss {train_loss:.4f}, acc {train_acc:.4f} | Val loss {val_loss:.4f}, acc {val_acc:.4f}")
        logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        checkpoint_path = f"checkpoints/epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    logging.info("Training complete. Saving model.")
    torch.save(model.state_dict(), Config.checkpoint_path)

if __name__ == '__main__':
    main()