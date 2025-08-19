# train.py
import torch
from tqdm import tqdm
import torch.nn.functional as F
from config import Config
from sklearn.metrics import precision_score, average_precision_score, f1_score, roc_auc_score, recall_score

def train(model, train_loader, optimizer, scheduler, criterion, epoch=0):
    model.train()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    loop = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

    for images, labels in loop:
        images, labels = images.to(Config.device), labels.to(Config.device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类概率
        preds = outputs.argmax(dim=1)

        # all_probs.extend(probs.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
        loop.set_postfix(loss=loss.item(), acc=acc)

    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    recall = recall_score(all_labels, all_preds)

    return total_loss / len(train_loader.dataset), acc, precision, ap, f1, auc, recall