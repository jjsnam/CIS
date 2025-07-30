import torch
import torch.nn as nn
import torchvision.models as models


class MultiRegionRCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.shared_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.shared_model = nn.Sequential(*list(self.shared_model.children())[:-1])  # remove fc
        self.feature_dim = 512

        # Classifier heads for each region
        self.region_heads = nn.ModuleDict({
            "full": nn.Linear(self.feature_dim, num_classes),
            "eyes": nn.Linear(self.feature_dim, num_classes),
            "mouth": nn.Linear(self.feature_dim, num_classes),
        })

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(self.feature_dim * 3, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, regions):
        feats = []
        for region in ["full", "eyes", "mouth"]:
            x = self.shared_model(regions[region]).squeeze()
            feats.append(x)
        concat = torch.cat(feats, dim=1)
        out = self.fusion(concat)
        return out
