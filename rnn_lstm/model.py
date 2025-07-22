import torch
import torch.nn as nn
from torchvision import models

class CNN_BiLSTM(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=1, bidirectional=True):
        super().__init__()
        base = models.resnet18(pretrained=True)
        modules = list(base.children())[:-1]  # 去掉最后一层FC
        self.feature_extractor = nn.Sequential(*modules)  # 输出维度 512
        self.feature_dim = 512
        self.lstm = nn.LSTM(input_size=self.feature_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=bidirectional)
        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.classifier = nn.Linear(lstm_out_dim, 2)

    def forward(self, image_seq):
        # image_seq: (batch, seq_len, C, H, W)
        batch_size, seq_len, C, H, W = image_seq.size()
        x = image_seq.view(batch_size * seq_len, C, H, W)
        x = self.feature_extractor(x).squeeze()  # (B*S, 512, 1, 1) → (B*S, 512)
        x = x.view(batch_size, seq_len, -1)  # (B, S, 512)
        lstm_out, _ = self.lstm(x)  # (B, S, H*2)
        last_hidden = lstm_out[:, -1, :]  # 取最后一步输出
        out = self.classifier(last_hidden)
        return out
