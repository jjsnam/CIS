import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def print_title(title):
    """打印带边框的标题"""
    print(f"\n{'=' * 50}")
    print(f"{title.upper()}")
    print(f"{'=' * 50}")


def load_data():
    """加载特征数据并显示进度"""
    print_title("加载特征数据")

    # 检查文件是否存在
    if not os.path.exists('statistical_features.npy'):
        print("错误: 特征文件 'statistical_features.npy' 不存在")
        print("请先运行 ela_features.py 提取特征")
        exit(1)

    # 加载特征
    print("加载特征矩阵...")
    features = np.load('statistical_features.npy')
    print(f"特征矩阵维度: {features.shape}")

    # 加载标签
    print("加载标签...")
    labels = np.load('statistical_labels.npy')
    print(f"标签数量: {len(labels)}")

    # 计算类别分布
    unique, counts = np.unique(labels, return_counts=True)
    class_names = {0: "真实图像", 1: "伪造图像"}
    print("\n类别分布:")
    for cls, count in zip(unique, counts):
        print(f"- {class_names[cls]}: {count} 张 ({count / len(labels) * 100:.1f}%)")

    return features, labels


def preprocess_data(features, labels):
    """数据预处理"""
    print_title("数据预处理")

    # 标准化特征
    print("标准化特征...")
    scaler = StandardScaler()

    # 添加进度条
    with tqdm(total=features.shape[1], desc="标准化进度") as pbar:
        # 逐列标准化并更新进度
        for i in range(features.shape[1]):
            col = features[:, i].reshape(-1, 1)
            scaler.partial_fit(col)
            pbar.update(1)

    # 应用标准化
    print("应用标准化转换...")
    features_scaled = np.zeros_like(features)
    with tqdm(total=features.shape[1], desc="应用转换") as pbar:
        for i in range(features.shape[1]):
            col = features[:, i].reshape(-1, 1)
            features_scaled[:, i] = scaler.transform(col).flatten()
            pbar.update(1)

    # 划分数据集
    print("\n划分训练集和测试集...")
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )

    print(f"训练集大小: {len(X_train)} 样本")
    print(f"测试集大小: {len(X_test)} 样本")

    return X_train, X_test, y_train, y_test, scaler


def train_svm(X_train, y_train):
    """训练SVM模型"""
    print_title("训练SVM模型")
    print("使用RBF核函数的支持向量机...")

    # 训练模型
    svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, verbose=False)

    start_time = time.time()
    svm.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"训练完成! 耗时: {train_time:.2f}秒")
    return svm


def train_random_forest(X_train, y_train):
    """训练随机森林模型"""
    print_title("训练随机森林模型")
    print("使用100棵树的随机森林分类器...")

    # 训练模型
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, verbose=1, n_jobs=-1)

    start_time = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - start_time

    print(f"训练完成! 耗时: {train_time:.2f}秒")
    return rf


def evaluate_model(model, X_test, y_test, model_name):
    """评估模型性能"""
    print_title(f"评估{model_name}模型")

    # 预测测试集
    print("预测测试集...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    print(f"预测完成! 耗时: {predict_time:.2f}秒")
    print(f"每秒预测: {len(X_test) / predict_time:.1f}张图像")

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n准确率: {accuracy:.4f}")

    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=["真实图像", "伪造图像"]))

    # 混淆矩阵
    print("混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["真实图像", "伪造图像"],
                yticklabels=["真实图像", "伪造图像"])
    plt.title(f'{model_name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.close()
    print(f"混淆矩阵已保存至: confusion_matrix_{model_name}.png")

    return accuracy


def visualize_feature_importance(rf, feature_count):
    """可视化特征重要性"""
    print_title("特征重要性分析")

    # 获取特征重要性
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    print("前10个重要特征:")
    for i in range(min(10, feature_count)):
        print(f"{i + 1}. 特征 {indices[i]}: {importances[indices[i]]:.4f}")

    # 可视化特征重要性
    plt.figure(figsize=(12, 8))
    plt.title("特征重要性")
    plt.bar(range(feature_count), importances[indices], align="center")
    plt.xticks(range(feature_count), indices)
    plt.xlim([-1, feature_count])
    plt.xlabel("特征索引")
    plt.ylabel("重要性")
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    print("特征重要性图已保存至: feature_importance.png")


if __name__ == "__main__":
    # 记录总开始时间
    total_start = time.time()

    # 1. 加载数据
    features, labels = load_data()

    # 2. 预处理数据
    X_train, X_test, y_train, y_test, scaler = preprocess_data(features, labels)

    # 3. 训练和评估SVM模型
    svm_model = train_svm(X_train, y_train)
    svm_accuracy = evaluate_model(svm_model, X_test, y_test, "SVM")

    # 4. 训练和评估随机森林模型
    rf_model = train_random_forest(X_train, y_train)
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, "RandomForest")

    # 5. 特征重要性分析
    visualize_feature_importance(rf_model, features.shape[1])

    # 最终报告
    print_title("最终结果总结")
    print(f"SVM模型准确率: {svm_accuracy:.4f}")
    print(f"随机森林模型准 确率: {rf_accuracy:.4f}")

    # 计算总耗时
    total_time = time.time() - total_start
    mins, secs = divmod(total_time, 60)
    print(f"\n总耗时: {int(mins)}分{secs:.1f}秒")

    # 保存模型
    import joblib

    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n模型已保存:")
    print("- svm_model.pkl")
    print("- random_forest_model.pkl")
    print("- scaler.pkl (标准化器)")