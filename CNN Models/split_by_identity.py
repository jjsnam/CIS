import os
import shutil
import random
from collections import defaultdict
from tqdm import tqdm

# 原始路径
DATA_ROOT = "/root/Project/datasets/Celeb_V2"
REAL_DIR = os.path.join(DATA_ROOT, "Train", "real")
FAKE_DIR = os.path.join(DATA_ROOT, "Train", "fake")

# 输出路径（身份不重叠）
NEW_ROOT = os.path.join(DATA_ROOT, "IdentitySplit")
TRAIN_REAL = os.path.join(NEW_ROOT, "Train", "real")
TRAIN_FAKE = os.path.join(NEW_ROOT, "Train", "fake")
VAL_REAL = os.path.join(NEW_ROOT, "Val", "real")
VAL_FAKE = os.path.join(NEW_ROOT, "Val", "fake")

def clean_and_create_dirs():
    for path in [TRAIN_REAL, TRAIN_FAKE, VAL_REAL, VAL_FAKE]:
        os.makedirs(path, exist_ok=True)

def collect_real_identities():
    identities = defaultdict(list)
    for fname in os.listdir(REAL_DIR):
        if fname.endswith(".jpg"):
            identity = fname[:5]  # e.g., "00000"
            identities[identity].append(fname)
    return identities

def collect_fake_identities():
    identities = defaultdict(list)
    for fname in os.listdir(FAKE_DIR):
        if fname.endswith(".jpg"):
            identity = fname.split("_")[0]  # e.g., "id0"
            identities[identity].append(fname)
    return identities

def split_identity_groups(id_dict, val_ratio=0.2):
    all_ids = list(id_dict.keys())
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * val_ratio)
    return set(all_ids[n_val:]), set(all_ids[:n_val])  # train_ids, val_ids

def copy_files_by_id(id_set, id_dict, src_folder, dst_folder):
    for identity in tqdm(id_set, desc=f"Copying to {dst_folder}"):
        for fname in id_dict[identity]:
            src = os.path.join(src_folder, fname)
            dst = os.path.join(dst_folder, fname)
            shutil.copy(src, dst)

def main():
    print("🔄 Cleaning & setting up directories...")
    clean_and_create_dirs()

    print("📂 Collecting identities...")
    real_ids = collect_real_identities()
    fake_ids = collect_fake_identities()

    real_train, real_val = split_identity_groups(real_ids)
    fake_train, fake_val = split_identity_groups(fake_ids)

    print("📥 Copying training/validation images (real)...")
    copy_files_by_id(real_train, real_ids, REAL_DIR, TRAIN_REAL)
    copy_files_by_id(real_val, real_ids, REAL_DIR, VAL_REAL)

    print("📥 Copying training/validation images (fake)...")
    copy_files_by_id(fake_train, fake_ids, FAKE_DIR, TRAIN_FAKE)
    copy_files_by_id(fake_val, fake_ids, FAKE_DIR, VAL_FAKE)

    print("✅ 数据集按身份划分完毕！")

if __name__ == "__main__":
    main()