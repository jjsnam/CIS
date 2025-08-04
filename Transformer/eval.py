import torch
from config import Config
from sklearn.metrics import precision_score, average_precision_score, f1_score, roc_auc_score, recall_score

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(Config.device), labels.to(Config.device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1]  # 取正类概率
            preds = outputs.argmax(dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
    precision = precision_score(all_labels, all_preds)
    ap = average_precision_score(all_labels, all_probs)
    f1 = f1_score(all_labels, all_preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.0
    recall = recall_score(all_labels, all_preds)

    return total_loss / len(val_loader), acc, precision, ap, f1, auc, recall