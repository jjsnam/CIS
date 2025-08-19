import os
import random
import shutil
from pathlib import Path

def sample_and_copy(src_dirs, dst_root, split_ratio=1/3, seed=42):
    random.seed(seed)
    splits = ["Train", "Val", "Test"]
    classes = ["real", "fake"]

    for split in splits:
        for cls in classes:
            dst_dir = Path(dst_root) / split / cls
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src_root in src_dirs:
                src_dir = Path(src_root) / split / cls
                if not src_dir.exists():
                    print(f"⚠️ Warning: {src_dir} not found, skip")
                    continue

                files = list(src_dir.glob("*"))
                n_select = len(files) // 3   # 取 1/3
                sampled = random.sample(files, n_select) if n_select > 0 else []

                print(f"[{split}/{cls}] {src_dir} -> {len(sampled)} images")

                for f in sampled:
                    dst_file = dst_dir / f.name
                    # 如果不同数据集有同名文件，加前缀避免冲突
                    if dst_file.exists():
                        dst_file = dst_dir / (src_dir.parent.parent.name + "_" + f.name)
                    shutil.copy(f, dst_file)


if __name__ == "__main__":
    # === 修改这里：填入 3 个数据集的路径 ===
    dataset1 = "/root/Project/datasets/200kMDID"
    dataset2 = "/root/Project/datasets/OpenForensics"
    dataset3 = "/root/Project/datasets/SGDF"

    # 融合后的目标路径
    mixed_dataset = "/root/Project/datasets/Fused"

    sample_and_copy([dataset1, dataset2, dataset3], mixed_dataset)
    print("✅ 数据集融合完成！")