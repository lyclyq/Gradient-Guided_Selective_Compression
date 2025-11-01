# models/small_cnn.py
import torch.nn as nn
from utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("small_cnn")
class SmallCNN(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 10, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28->14, 32->16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 14->7, 16->8
        )
        # 统一到 7x7，兼容 MNIST(28) & CIFAR(32)
        self.adapt = nn.AdaptiveAvgPool2d(7)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(64 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.adapt(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)
