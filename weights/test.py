import os
import argparse
from glob import glob
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import numpy as np

# 假设你已有 face_extractor.py, RCNN 模型等在同一目录
from face_extractor import extract_features
from models import CNNModel, RCNNModel, TransformerModel, CNNTransformerModel  # 根据你的 repo 调整导入

# ------------------------
# Dataset 类
# ------------------------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        classes = ['real', 'fake']
        for idx, cls in enumerate(classes):
            cls_dir = os.path.join(root_dir, cls)
            for img_path in glob(os.path.join(cls_dir, '*')):
                self.samples.append(img_path)
                self.labels.append(idx)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ------------------------
# 测试函数
# ------------------------
def test_model(model_type, model_path, test_path, batch_size=32, device='cuda'):
    # ------------------------
    # 数据准备
    # ------------------------
    if model_type.lower() != 'gmm':
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 假设训练时为 224x224
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])
        dataset = ImageFolderDataset(test_path, transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        # GMM 直接提取特征
        dataset = ImageFolderDataset(test_path)
    
    y_true = []
    y_pred = []
    
    # ------------------------
    # 模型加载
    # ------------------------
    if model_type.lower() == 'cnn':
        model = CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
    elif model_type.lower() == 'rcnn':
        model = RCNNModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
    elif model_type.lower() == 'transformer':
        model = TransformerModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
    elif model_type.lower() == 'cnn+transformer':
        model = CNNTransformerModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device).eval()
    elif model_type.lower() == 'gmm':
        model = joblib.load(model_path)
    else:
        raise ValueError("Unsupported model type: {}".format(model_type))
    
    # ------------------------
    # 推理
    # ------------------------
    if model_type.lower() != 'gmm':
        with torch.no_grad():
            for imgs, labels in tqdm(loader):
                imgs = imgs.to(device)
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                y_pred.extend(preds)
                y_true.extend(labels.numpy())
    else:
        # GMM 特征预测
        for img, label in tqdm(dataset):
            feat = extract_features(img)  # 返回 numpy array
            pred = model.predict([feat])[0]
            y_pred.append(pred)
            y_true.append(label)
    
    # ------------------------
    # 指标计算
    # ------------------------
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        roc = roc_auc_score(y_true, y_pred)
    except:
        roc = float('nan')
    
    print(f"Model Type: {model_type}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

# ------------------------
# 命令行入口
# ------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True, help='cnn / rcnn / transformer / cnn+transformer / gmm')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    test_model(args.model_type, args.model_path, args.test_path, args.batch_size, args.device)
#python test.py --model_type cnn --model_path /root/Project/