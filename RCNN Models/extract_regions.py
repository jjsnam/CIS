# extract_regions.py

import os
import shutil
from tqdm import tqdm
from face_detector import FaceRegionExtractor

# ===== 配置部分 =====
dataset_name = "Celeb_V2"
splits = ["Train", "Val", "Test"]
base_image_root = f"/root/Project/datasets/{dataset_name}"
base_cache_dir = "/root/Project/RCNN Models/cache/regions"

# ===== 初始化提取器 =====
extractor = FaceRegionExtractor(cache_dir=base_cache_dir)

# ===== 清除旧缓存并处理每个 split =====
for split in splits:
    image_root = os.path.join(base_image_root, split)
    cache_root = os.path.join(base_cache_dir, dataset_name, split)

    print(f"\n>>> Processing split: {split}")
    if os.path.exists(cache_root):
        print(f"[INFO] Removing old cache at {cache_root}")
        shutil.rmtree(cache_root)

    for cls in ["real", "fake"]:
        img_dir = os.path.join(image_root, cls)
        if not os.path.exists(img_dir):
            print(f"[WARN] Directory not found: {img_dir}")
            continue

        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
        print(f"Processing {cls} images, total: {len(img_paths)}")

        for path in tqdm(img_paths):
            try:
                regions = extractor.extract(path)
                for region_type, region_img in regions.items():
                    save_dir = os.path.join(cache_root, cls, region_type)
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.basename(path)
                    save_path = os.path.join(save_dir, filename)
                    region_img.save(save_path)
            except Exception as e:
                print(f"[SKIP] Failed to process {path}: {e}")