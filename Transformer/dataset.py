# # dataset.py
# from torchvision import transforms, datasets
# from torch.utils.data import DataLoader
# from config import Config

# def get_dataloaders():
#     # transform = transforms.Compose([
#     #     transforms.Resize((Config.image_size, Config.image_size)),
#     #     transforms.ToTensor(),
#     #     transforms.Normalize([0.5], [0.5])
#     # ])
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(Config.image_size, scale=(0.8, 1.0)),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.RandomRotation(degrees=10),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])

#     val_transform = transforms.Compose([
#         transforms.Resize((Config.image_size, Config.image_size)),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5], [0.5])
#     ])

#     train_ds = FFTImageFolder(root='/root/Project/datasets/OpenForensics/Train', transform=train_transform)
#     val_ds = FFTImageFolder(root='/root/Project/datasets/OpenForensics/Val', transform=val_transform)

#     # train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
#     # val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False)
#     train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, val_loader


# # --- FFTImageFolder definition ---
# from PIL import Image
# import numpy as np
# import torch
# import os

# class FFTImageFolder(datasets.ImageFolder):
#     def __getitem__(self, index):
#         path, target = self.samples[index]
#         img = Image.open(path).convert('RGB')
#         if self.transform is not None:
#             img_rgb = self.transform(img)

#             # 计算 FFT 幅度图
#             np_img = np.array(img) / 255.0
#             fft = np.fft.fft2(np_img, axes=(0, 1))
#             fft_shift = np.fft.fftshift(fft, axes=(0, 1))
#             magnitude = np.abs(fft_shift)
#             magnitude = np.log1p(magnitude)  # 取对数
#             magnitude = magnitude / magnitude.max()

#             fft_tensor = torch.tensor(magnitude.transpose(2, 0, 1), dtype=torch.float32)

#             # 拼接 RGB + FFT
#             img_combined = torch.cat([img_rgb, fft_tensor], dim=0)
#             return img_combined, target
#         return img, target
# dataset.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config import Config

def get_dataloaders(train_dir, val_dir):
    train_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    # train_transform = transforms.Compose([
    #     transforms.RandomResizedCrop(Config.image_size, scale=(0.8, 1.0)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    #     transforms.RandomRotation(degrees=10),
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5], [0.5])
    # ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # train_ds = datasets.ImageFolder(root='/root/Project/datasets/SGDF/Train', transform=train_transform)
    train_ds = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_transform)

    # train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False)
    train_loader = DataLoader(train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader