import torch
from config import Config
# eval.py
def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    acc = correct / len(val_loader.dataset)
    return total_loss / len(val_loader), acc