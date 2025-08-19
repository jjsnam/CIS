import os
from PIL import Image
from torch.utils.data import Dataset
import torch


class RegionFaceDataset(Dataset):
    def __init__(self, root_dir, cache_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.cache_dir = cache_dir
        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root_dir, cls)
            # print(cls_dir)
            for fname in os.listdir(cls_dir):
                if not fname.endswith(".jpg"):
                    continue
                img_path = os.path.join(cls_dir, fname)
                # print(fname)
                basename = fname.split(".")[0]
                region_paths = {
                    "full": os.path.join(self.cache_dir, cls, "full", f"{basename}.jpg"),
                    "eyes": os.path.join(self.cache_dir, cls, "eyes", f"{basename}.jpg"),
                    "mouth": os.path.join(self.cache_dir, cls, "mouth", f"{basename}.jpg"),
                }
                if all(os.path.exists(p) for p in region_paths.values()):
                    self.samples.append((img_path, label))
                else:
                    # print(f"[WARN] Skip {img_path}, region image missing.")
                    # print(f"{region_paths}")
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import warnings
        max_retries = 10
        retries = 0

        while retries < max_retries:
            img_path, label = self.samples[idx]
            fname = os.path.basename(img_path)
            if "real" in img_path:
                class_name = "real"
                basename = fname.split(".")[0]
            else:
                class_name = "fake"
                basename = fname.split(".")[0]
            region_paths = {
                "full": os.path.join(self.cache_dir, class_name, "full", f"{basename}.jpg"),
                "eyes": os.path.join(self.cache_dir, class_name, "eyes", f"{basename}.jpg"),
                "mouth": os.path.join(self.cache_dir, class_name, "mouth", f"{basename}.jpg"),
            }

            regions = {}
            try:
                for k, path in region_paths.items():
                    img = Image.open(path).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    regions[k] = img
                return regions, torch.tensor(label, dtype=torch.long)
            except Exception as e:
                retries += 1
                idx = int(torch.randint(0, len(self.samples), (1,)).item())

        warnings.warn(f"Failed to load a valid sample after {max_retries} retries.")
        raise RuntimeError(f"Failed to load a valid sample after {max_retries} retries.")
