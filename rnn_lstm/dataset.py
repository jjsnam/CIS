import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=5, transform=None, repeat_per_identity=5):
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []  # list of (sequence_image_paths, label)

        for label, cls in enumerate(["real", "fake"]):
            cls_dir = os.path.join(root_dir, cls)
            identity_dict = {}

            for fname in os.listdir(cls_dir):
                if not fname.endswith(".jpg"):
                    continue
                if cls == "real":
                    identity = fname.split("_")[0]
                else:
                    identity = fname.split("_")[0] + "_" + fname.split("_")[1]

                identity_dict.setdefault(identity, []).append(os.path.join(cls_dir, fname))

            for identity, files in identity_dict.items():
                if len(files) >= sequence_length:
                    for _ in range(min(repeat_per_identity, len(files) // sequence_length)):
                        self.samples.append((files, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths, label = self.samples[idx]
        chosen = random.sample(paths, self.sequence_length)
        images = []

        for path in chosen:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # shape: (sequence_length, C, H, W)
        return images, label