# import os
# import random
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms

# class SequenceDataset(Dataset):
#     def __init__(self, root_dir, sequence_length=5, transform=None, repeat_per_identity=5):
#         self.sequence_length = sequence_length
#         self.transform = transform
#         self.samples = []  # list of (sequence_image_paths, label)

#         for label, cls in enumerate(["real", "fake"]):
#             cls_dir = os.path.join(root_dir, cls)
#             identity_dict = {}

#             for fname in os.listdir(cls_dir):
#                 if not fname.endswith(".jpg"):
#                     continue
#                 if cls == "real":
#                     identity = fname.split("_")[0]
#                 else:
#                     identity = fname.split("_")[0] + "_" + fname.split("_")[1]

#                 identity_dict.setdefault(identity, []).append(os.path.join(cls_dir, fname))

#             for identity, files in identity_dict.items():
#                 if len(files) >= sequence_length:
#                     for _ in range(min(repeat_per_identity, len(files) // sequence_length)):
#                         self.samples.append((files, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         paths, label = self.samples[idx]
#         chosen = random.sample(paths, self.sequence_length)
#         images = []

#         for path in chosen:
#             img = Image.open(path).convert("RGB")
#             if self.transform:
#                 img = self.transform(img)
#             images.append(img)

#         # shape: (sequence_length, C, H, W)
#         return images, label
# dataset.py
# import os
# import random
# from PIL import Image
# from torch.utils.data import Dataset
# import torch

# class SequenceDataset(Dataset):
#     def __init__(self, root_dir, sequence_length=5, transform=None, repeat_per_identity=5):
#         self.sequence_length = sequence_length
#         self.transform = transform
#         self.samples = []

#         for label, cls in enumerate(["real", "fake"]):
#             cls_dir = os.path.join(root_dir, cls)
#             identity_dict = {}

#             for fname in os.listdir(cls_dir):
#                 if not fname.endswith(".jpg"):
#                     continue
#                 if cls == "real":
#                     identity = fname.split("_")[0]
#                 else:
#                     identity = fname.split("_")[0] + "_" + fname.split("_")[1]

#                 identity_dict.setdefault(identity, []).append(os.path.join(cls_dir, fname))

#             for identity, files in identity_dict.items():
#                 if len(files) >= sequence_length:
#                     for _ in range(min(repeat_per_identity, len(files) // sequence_length)):
#                         self.samples.append((files, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         paths, label = self.samples[idx]
#         chosen = random.sample(paths, self.sequence_length)
#         images = []

#         for path in chosen:
#             img = Image.open(path).convert("RGB")
#             if self.transform:
#                 img = self.transform(img)
#             images.append(img)

#         # Stack to (T, C, H, W)
#         images = torch.stack(images)
#         labels = torch.tensor([label] * self.sequence_length)  # (T,)

#         return images, labels
import os
import random
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms
import hashlib

class SequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, repeat_per_identity=5):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root_dir, cls)
            identity_dict = {}

            for fname in os.listdir(cls_dir):
                if not fname.endswith(".jpg"):
                    continue
                path = os.path.join(cls_dir, fname)

                # 尝试从文件名提取 identity，否则 fallback 为路径哈希
                try:
                    if cls == "real":
                        identity = fname.split("_")[0]
                    else:
                        identity = fname.split("_")[0] + "_" + fname.split("_")[1]
                except IndexError:
                    identity = hashlib.md5(path.encode()).hexdigest()

                identity_dict.setdefault(identity, []).append(path)

            for identity, files in identity_dict.items():
                if len(files) >= sequence_length:
                    for _ in range(min(repeat_per_identity, len(files) // sequence_length)):
                        self.samples.append((files, label))

        if len(self.samples) < 10:
            print(f"⚠️ Warning: Only {len(self.samples)} usable sequences were found. "
                  f"Check file naming and dataset structure.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        chosen = random.sample(paths, self.sequence_length)
        images = []

        for path in chosen:
            try:
                img = Image.open(path).convert("RGB")
                if self.transform:
                    img = self.transform(img)
                images.append(img)
            except (UnidentifiedImageError, OSError) as e:
                print(f"⚠️ Skipping unreadable image: {path} — {e}")
                return self.__getitem__((idx + 1) % len(self))

        return images, torch.tensor([label] * self.sequence_length)