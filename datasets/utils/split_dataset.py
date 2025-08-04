# for SGDF dataset
import os
import shutil
import random
from pathlib import Path

# 原始路径（包含 fake/ 和 real/）
original_root = 'dataset/__SGDF'  # ← 你原始目录
output_root = 'dataset/SGDF'

classes = ['Real', 'Fake']
split_ratio = (0.7, 0.15, 0.15)  # train, val, test

random.seed(42)

# 创建目标目录
for split in ['Train', 'Val', 'Test']:
    for cls in classes:
        Path(f'{output_root}/{split}/{cls}').mkdir(parents=True, exist_ok=True)

# 开始划分
for cls in classes:
    files = os.listdir(os.path.join(original_root, cls))
    files = [f for f in files if not f.startswith('.')]  # 忽略隐藏文件
    random.shuffle(files)

    total = len(files)
    n_train = int(total * split_ratio[0])
    n_val = int(total * split_ratio[1])

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    print(f"{cls.upper()}: Total={total}, Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

    # 拷贝文件
    for split, split_files in zip(['Train', 'Val', 'Test'], [train_files, val_files, test_files]):
        for fname in split_files:
            src_path = os.path.join(original_root, cls, fname)
            dst_path = os.path.join(output_root, split, cls, fname)
            shutil.copy(src_path, dst_path)

print("✅ 数据集划分完成！")