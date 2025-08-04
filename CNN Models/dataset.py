# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DATAset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.jpg') or file.endswith('.png'):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label