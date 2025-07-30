# train.py
import torch
from tqdm import tqdm
import torch.nn.functional as F
from config import Config

def train(model, train_loader, optimizer, scheduler, criterion, epoch=None):
    model.train()
    total_loss = 0
    correct = 0

    loop = tqdm(train_loader, desc=f"Training Epoch {epoch}")

    for images, labels in loop:
        images, labels = images.to(Config.device), labels.to(Config.device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        loop.set_postfix(loss=loss.item(), acc=correct / ((loop.n + 1e-8) * Config.batch_size))

    acc = correct / len(train_loader.dataset)
    return total_loss / len(train_loader.dataset), acc