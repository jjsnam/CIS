import torch
import torch.nn as nn
from torchvision.models import resnet18
from timm.models import create_model
import os


class CNNTransformerClassifier(nn.Module):
    def __init__(self, transformer_name='vit_base_patch16_224', pretrained=True, num_classes=2):
        super(CNNTransformerClassifier, self).__init__()

        # CNN Backbone (ResNet18)
        resnet = resnet18(pretrained=True)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-3])  # Output: (B, 256, 28, 28)

        # Project CNN feature map to ViT patch embedding size
        self.proj = nn.Conv2d(256, 768, kernel_size=1)  # match ViT hidden dim (768)

        # ViT Transformer
        self.transformer = create_model(
            transformer_name,
            pretrained=False,
            num_classes=0  # Do not load classifier head
        )

        # Load ViT weights excluding classifier head
        vit_ckpt = '/root/Project/Transformer/weights/vit_base_patch16_224.pth'
        state_dict = torch.load(vit_ckpt, map_location='cpu')
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        missing_keys, unexpected_keys = self.transformer.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded ViT weights. Missing: {missing_keys}, Unexpected: {unexpected_keys}")

        # Freeze ViT classifier head (we use our own)
        self.transformer.reset_classifier(0)

        # Classification Head
        self.cls_head = nn.Linear(768, num_classes)

    def forward(self, x):
        feat = self.cnn_backbone(x)  # (B, 256, 28, 28)
        feat = self.proj(feat)       # (B, 768, 28, 28)

        B, C, H, W = feat.shape
        feat = feat.flatten(2).transpose(1, 2)  # (B, HW, C)

        # Add CLS token manually
        cls_token = self.transformer.cls_token.expand(B, -1, -1)
        feat = torch.cat((cls_token, feat), dim=1)  # (B, 1+HW, C)

        # Add positional embedding
        pos_embed = self.transformer.pos_embed[:, :feat.size(1), :]
        feat = feat + pos_embed

        feat = self.transformer.blocks(feat)
        feat = self.transformer.norm(feat)

        cls_output = feat[:, 0]  # CLS token
        out = self.cls_head(cls_output)
        return out