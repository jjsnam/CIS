import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from tqdm import tqdm
# from thundersvm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import torch

def load_images_from_folder(folder_path, label, img_size=(64, 64)):
    data = []
    labels = []
    for filename in tqdm(os.listdir(folder_path), desc=f"Loading {label} from {os.path.basename(folder_path)}"):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = cv2.imread(img_path)  # BGR 图像
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为 RGB
                img_resized = cv2.resize(img, img_size)
                img_flatten = img_resized.flatten()
                data.append(img_flatten)
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(data), np.array(labels)

def load_dataset(base_path, img_size=(64, 64)):
    sets = ['Train', 'Val', 'Test']
    data = {}
    for set_name in sets:
        real_path = os.path.join(base_path, set_name, 'real')
        fake_path = os.path.join(base_path, set_name, 'fake')

        X_real, y_real = load_images_from_folder(real_path, 'real', img_size)
        X_fake, y_fake = load_images_from_folder(fake_path, 'fake', img_size)

        X = np.concatenate([X_real, X_fake], axis=0)
        y = np.concatenate([y_real, y_fake], axis=0)

        data[set_name] = (X, y)
    return data

dataset_path = r'/root/Project/datasets/Celeb_V2'  # 改成你本地的路径
data = load_dataset(dataset_path, img_size=(32, 32))

X_train, y_train = data['Train']
X_val, y_val = data['Val']
X_test, y_test = data['Test']

# 降低内存消耗，转为 float32
X_train = X_train.astype(np.float32)
X_val = X_val.astype(np.float32)
X_test = X_test.astype(np.float32)


# 使用 PyTorch GPU PCA 降维
def pca_torch(X_numpy, k=128):
    X = torch.from_numpy(X_numpy).float().cuda()
    X = X - torch.mean(X, dim=0, keepdim=True)
    U, S, V = torch.linalg.svd(X, full_matrices=False)
    X_reduced = torch.matmul(X, V[:k].T)
    return X_reduced.cpu().numpy()

X_train = pca_torch(X_train, k=128)
X_val = pca_torch(X_val, k=128)
X_test = pca_torch(X_test, k=128)

encoder = LabelEncoder()
y_train_enc = encoder.fit_transform(y_train)
y_val_enc = encoder.transform(y_val)
y_test_enc = encoder.transform(y_test)

clf = SGDClassifier(loss='hinge',
                   max_iter=1000,
                   tol=1e-3,
                   early_stopping=True,
                   validation_fraction=0.1,
                   n_iter_no_change=5,
                   verbose=1,
                   random_state=42)

X_train, y_train_enc = shuffle(X_train, y_train_enc, random_state=42)

print("开始训练 SGDClassifier 模型...")
from time import time
start_time = time()
try:
    clf.fit(X_train, np.array(y_train_enc))
except KeyboardInterrupt:
    print("检测到中断信号，正在保存中间模型...")
    joblib.dump(clf, "models/sgd_model_cv2_partial.joblib")
    print("部分模型已保存为 models/sgd_model_cv2_partial.joblib")
    exit()
print(f"SGDClassifier 训练完成，总耗时 {time() - start_time:.1f} 秒")

# 验证集性能
y_val_pred = clf.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val_enc, y_val_pred))

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/sgd_model_cv2.joblib")
joblib.dump(encoder, "models/label_encoder_cv2.joblib")
print("模型和标签编码器已保存。")

loaded_clf = joblib.load("models/sgd_model_cv2.joblib")
loaded_encoder = joblib.load("models/label_encoder_cv2.joblib")

y_test_pred = loaded_clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test_enc, y_test_pred))
print("Classification Report:\n", classification_report(y_test_enc, y_test_pred, target_names=loaded_encoder.classes_))

def plot_sample_predictions(y_true, y_pred, n=10):
    from matplotlib import pyplot as plt
    correct = (y_true == y_pred)
    indices = np.arange(len(y_true))

    plt.figure(figsize=(10, 2))
    plt.bar(indices[:n], correct[:n], color=["green" if c else "red" for c in correct[:n]])
    plt.xticks(indices[:n])
    plt.ylim(0, 1)
    plt.title("Prediction Correctness (Green=Correct, Red=Wrong)")
    plt.show()

plot_sample_predictions(y_test_enc, y_test_pred, n=20)
