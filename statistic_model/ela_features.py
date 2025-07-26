import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from skimage.color import rgb2hsv
from tqdm import tqdm  # 导入进度条库
import time  # 添加时间戳
import multiprocessing
from functools import partial

# def unwrap_process_image(args):
#     return _process_image(*args)

def extract_ela_features(image_path, quality=90):
    """提取ELA特征和基础统计特征"""
    try:
        # 读取并转换图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法读取图像 {image_path}")
            return None

        # ELA特征：重压缩差异
        temp_path = "temp.jpg"
        cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(temp_path)

        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)

        if compressed is None:
            print(f"警告: 无法读取压缩图像 {image_path}")
            return None

        ela_map = np.abs(img.astype(np.float32) - compressed.astype(np.float32))
        ela_map = ela_map.max(axis=2)  # 取三通道最大值

        # 基础统计特征
        features = {
            'ela_mean': np.mean(ela_map),
            'ela_std': np.std(ela_map),
            'ela_max': np.max(ela_map),
            'ela_entropy': -np.sum(ela_map * np.log2(ela_map + 1e-7))
        }

        # LBP纹理特征
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=59, range=(0, 58))
        features.update({f'lbp_{i}': val for i, val in enumerate(hist)})

        # 颜色特征
        hsv = rgb2hsv(img / 255.0)
        features['hue_std'] = np.std(hsv[:, :, 0])
        features['sat_std'] = np.std(hsv[:, :, 1])

        return list(features.values())
    except Exception as e:
        print(f"处理 {image_path} 时出错: {str(e)}")
        return None


def process_dataset(data_dir):
    """处理整个数据集"""
    features = []
    labels = []
    processed_count = 0
    error_count = 0

    # 获取开始时间
    start_time = time.time()

    print(f"\n{'=' * 50}")
    print(f"开始处理数据集: {data_dir}")
    print(f"{'=' * 50}\n")

    for label, folder in enumerate(['real', 'fake']):
        folder_path = os.path.join(data_dir, folder)

        # 检查目录是否存在
        if not os.path.exists(folder_path):
            print(f"错误: 目录不存在 - {folder_path}")
            continue

        # 获取图像列表
        img_list = os.listdir(folder_path)
        if not img_list:
            print(f"警告: 目录为空 - {folder_path}")
            continue

        print(f"处理 {folder} 图像 ({len(img_list)} 张)...")

        # 添加进度条
        for img_name in tqdm(img_list, desc=folder, unit="img"):
            img_path = os.path.join(folder_path, img_name)

            # 只处理图像文件
            if not os.path.isfile(img_path) or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            feat = extract_ela_features(img_path)
            if feat is not None:
                features.append(feat)
                labels.append(label)
                processed_count += 1
            else:
                error_count += 1

    # 计算总耗时
    elapsed_time = time.time() - start_time

    print(f"\n{'=' * 50}")
    print(f"处理完成! 总耗时: {elapsed_time:.2f} 秒")
    print(f"成功处理: {processed_count} 张图片")
    print(f"失败处理: {error_count} 张图片")
    print(f"提取特征维度: {len(features[0]) if features else 0}")
    print(f"{'=' * 50}")

    return np.array(features), np.array(labels)

# def _process_image(img_path, label):
#     """用于多进程并行调用的图像处理函数"""
#     if not os.path.isfile(img_path) or not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
#         return None
#     feat = extract_ela_features(img_path)
#     if feat is not None:
#         return (feat, label)
#     return None

# def process_dataset(data_dir):
#     """并行处理整个数据集"""
#     features = []
#     labels = []
#     processed_count = 0
#     error_count = 0

#     start_time = time.time()

#     print(f"\n{'=' * 50}")
#     print(f"开始处理数据集（并行加速）: {data_dir}")
#     print(f"{'=' * 50}\n")

#     tasks = []

#     for label, folder in enumerate(['real', 'fake']):
#         folder_path = os.path.join(data_dir, folder)
#         if not os.path.exists(folder_path):
#             print(f"错误: 目录不存在 - {folder_path}")
#             continue
#         img_list = os.listdir(folder_path)
#         print(f"收集 {folder} 图像路径 ({len(img_list)} 张)...")
#         for img_name in img_list:
#             img_path = os.path.join(folder_path, img_name)
#             tasks.append((img_path, label))

#     print(f"共计 {len(tasks)} 张图像，开始并行处理...")

#     with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#         # for result in tqdm(pool.imap_unordered(lambda args: _process_image(*args), tasks), total=len(tasks)):
#         for result in tqdm(pool.imap_unordered(unwrap_process_image, tasks), total=len(tasks)):
#             if result is not None:
#                 feat, label = result
#                 features.append(feat)
#                 labels.append(label)
#                 processed_count += 1
#             else:
#                 error_count += 1

#     elapsed_time = time.time() - start_time

#     print(f"\n{'=' * 50}")
#     print(f"处理完成! 总耗时: {elapsed_time:.2f} 秒")
#     print(f"成功处理: {processed_count} 张图片")
#     print(f"失败处理: {error_count} 张图片")
#     print(f"提取特征维度: {len(features[0]) if features else 0}")
#     print(f"{'=' * 50}")

#     return np.array(features), np.array(labels)


if __name__ == "__main__":
    # 设置数据路径 - 修改为你的实际路径
    DATA_DIR = r"/root/Project/data/IdentitySplit/Train"  # 使用相对路径

    # 检查目录是否存在
    if not os.path.exists(DATA_DIR):
        print(f"错误: 主数据目录不存在 - {DATA_DIR}")
        print("请创建以下目录结构:")
        print(f"{DATA_DIR}/")
        print(f"├── real/  # 存放真实图像")
        print(f"└── fake/  # 存放伪造图像")
        exit(1)

    # 检查子目录是否存在
    for subdir in ['real', 'fake']:
        subdir_path = os.path.join(DATA_DIR, subdir)
        if not os.path.exists(subdir_path):
            print(f"警告: 子目录不存在 - {subdir_path}")
            print("创建目录中...")
            os.makedirs(subdir_path, exist_ok=True)

    # 处理数据集
    features, labels = process_dataset(DATA_DIR)

    # 保存结果
    if len(features) > 0:
        np.save('statistical_features.npy', features)
        np.save('statistical_labels.npy', labels)
        print("\n结果已保存至:")
        print(f"- statistical_features.npy (特征数据, {features.shape[0]}x{features.shape[1]})")
        print(f"- statistical_labels.npy (标签数据, {len(labels)})")
    else:
        print("错误: 未提取到任何特征，请检查数据目录内容")