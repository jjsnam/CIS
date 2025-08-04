import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==== 📝 请修改下面的路径 ====
csv_dir = '/root/Project'  # 包含 train_labels.csv 等的目录
image_dir_real = '/root/Project/my_real_vs_ai_dataset/my_real_vs_ai_dataset/real'  # 原始 real 图像目录
image_dir_fake = '/root/Project/my_real_vs_ai_dataset/my_real_vs_ai_dataset/ai_images'  # 原始 fake 图像目录
output_root = '/root/Project/datasets/200kMDID'  # 输出数据集根目录
# =================================

# CSV文件及目标子目录的映射
splits = {
    "Train": "train_labels.csv",
    "Val": "val_labels.csv",
    "Test": "test_labels.csv"
}

# 为每个划分创建目标子目录
for split in splits:
    for cls in ['real', 'fake']:
        os.makedirs(os.path.join(output_root, split, cls), exist_ok=True)

# 复制图像
for split, csv_file in splits.items():
    df = pd.read_csv(os.path.join(csv_dir, csv_file))
    print(f"Processing {split} set: {len(df)} images")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        label = row['label']
        cls = 'real' if label == 1 else 'fake'
        src_dir = image_dir_real if cls == 'real' else image_dir_fake
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(output_root, split, cls, filename)

        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
        else:
            print(f"❌ File not found: {src_path}")