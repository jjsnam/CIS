import zipfile
import os

# 检查压缩包路径（根据实际路径修改）
# zip_path = '/content/drive/MyDrive/archive.zip'  # 如果在子文件夹中：'/content/drive/MyDrive/data/dataset.zip'

# 创建解压目标文件夹
# !mkdir -p /content/dataset  # 在Colab临时空间解压（速度快）

# 解压到/content/dataset
# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # zip_ref.extractall('/content/dataset')

# print("解压完成！文件列表：", os.listdir('/content/dataset'))
import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm
from sklearn.decomposition import PCA

def load_data_from_folder(folder_path, image_size=(32, 32)):
    X, y = [], []
    for label_name in ['real', 'fake']:
        label_dir = os.path.join(folder_path, label_name)
        label = 0 if label_name == 'real' else 1

        for fname in tqdm(os.listdir(label_dir), desc=f"Loading {label_name}"):
            fpath = os.path.join(label_dir, fname)
            img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, image_size)
                X.append(img.flatten())  # 将图像展平为1D向量
                y.append(label)
    return np.array(X), np.array(y)

BASE_DIR = r'/root/Project/datasets/Celeb_V2/IdentitySplit'

train_dir = os.path.join(BASE_DIR, 'Train')
test_dir  = os.path.join(BASE_DIR, 'Val')

X_train, y_train = load_data_from_folder(train_dir)
X_test, y_test   = load_data_from_folder(test_dir)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

pca = PCA(n_components=100)  # 可调整维度数
X_train_pca = pca.fit_transform(X_train)
X_test_pca  = pca.transform(X_test)

clf = LinearSVC()
clf.fit(X_train_pca, y_train)

# 预测
y_pred = clf.predict(X_test_pca)

# 评估
print("准确率：", accuracy_score(y_test, y_pred))
print("\n分类报告：\n", classification_report(y_test, y_pred))

import joblib

# 保存模型和 PCA
joblib.dump(clf, 'svc_model.pkl')
joblib.dump(pca, 'pca_model.pkl')