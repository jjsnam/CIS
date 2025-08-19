# test_rcnn.py
import argparse
import torch
from torch.utils.data import DataLoader
from dataset import RegionFaceDataset
from rcnn_model import MultiRegionRCNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

def test(model_path, test_root, cache_root, batch_size=32, device='cuda', visualize_cm=False):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Define transforms consistent with training (Resize + ToTensor)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Dataset
    test_dataset = RegionFaceDataset(
        test_root,
        cache_root,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = MultiRegionRCNN()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for regions, labels in test_loader:
            regions = {k: v.to(device) for k, v in regions.items()}
            labels = labels.to(device)

            outputs = model(regions)
            outputs = outputs.view(outputs.size(0), -1)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设二分类
            preds = torch.argmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    roc_auc = roc_auc_score(all_labels, all_probs)

    print(f"[TEST] Accuracy:  {acc:.4f}")
    print(f"[TEST] Precision: {precision:.4f}")
    print(f"[TEST] Recall:    {recall:.4f}")
    print(f"[TEST] F1 Score:  {f1:.4f}")
    print(f"[TEST] ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(all_labels, all_preds)
    print("[TEST] Confusion Matrix:")
    print(cm)

    if visualize_cm:
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real','Fake'], yticklabels=['Real','Fake'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test RCNN model for Deepfake detection')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained RCNN model checkpoint')
    parser.add_argument('--test_root', type=str, required=True, help='Path to test dataset root folder (real/fake)')
    parser.add_argument('--cache_root', type=str, required=True, help='Path to pre-extracted cached region images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--visualize_cm', action='store_true', help='Visualize the confusion matrix')
    args = parser.parse_args()

    test(args.model_path, args.test_root, args.cache_root, args.batch_size, args.device, args.visualize_cm)
    # python test.py --model_path /root/Project/weights/RCNN/SGDF/SGDF_RCNN_top10.pth --test_root /root/Project/datasets/SGDF/Test --cache_root /root/Project/RCNN\ Models/cache/regions/SGDF/Test