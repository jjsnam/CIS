import os
from pathlib import Path
import shutil
from tqdm import tqdm

def fuse_rcnn_features_strict(fused_dataset_dir, rcnn_source_dirs, fused_rcnn_dir):
    splits = ["Train", "Val", "Test"]
    classes = ["real", "fake"]
    subregions = ["eyes", "full", "mouth"]

    fused_dataset_dir = Path(fused_dataset_dir)
    fused_rcnn_dir = Path(fused_rcnn_dir)

    for split in splits:
        for cls in classes:
            src_images = list((fused_dataset_dir / split / cls).glob("*.jpg"))
            print(f"[{split}/{cls}] {len(src_images)} images to process")
            for img_path in tqdm(src_images, desc=f"Processing {split}/{cls}"):
                img_name = img_path.name
                found_all = False
                # 尝试在三个源数据集中查找完整三类裁剪
                for src_root in rcnn_source_dirs:
                    src_paths = [Path(src_root) / split / cls / sub / img_name for sub in subregions]
                    if all(p.exists() for p in src_paths):
                        # 全部存在，复制到目标
                        for sub, src_file in zip(subregions, src_paths):
                            dst_sub_dir = fused_rcnn_dir / split / cls / sub
                            dst_sub_dir.mkdir(parents=True, exist_ok=True)
                            shutil.copy(src_file, dst_sub_dir / img_name)
                        found_all = True
                        break  # 找到一个数据集完整即可，不再查找其他数据集
                if not found_all:
                    print(f"⚠️ Image {img_name} missing one or more subregions, skipped")

if __name__ == "__main__":
    # 融合后的原始图像路径
    fused_dataset_dir = "/root/Project/datasets/Fused"

    # 原始三个数据集 RCNN 特征路径
    rcnn_source_dirs = [
        "/root/Project/RCNN Models/cache/regions/200kMDID",
        "/root/Project/RCNN Models/cache/regions/SGDF",
        "/root/Project/RCNN Models/cache/regions/OpenForensics"
    ]

    # 融合后的 RCNN 目标路径
    fused_rcnn_dir = "/root/Project/RCNN Models/cache/regions/Fused"

    fuse_rcnn_features_strict(fused_dataset_dir, rcnn_source_dirs, fused_rcnn_dir)
    print("✅ 严格对齐 RCNN 数据融合完成！")