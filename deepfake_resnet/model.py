# model.py
import torch.nn as nn
from torchvision import models

def get_resnet(num_classes=2):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model