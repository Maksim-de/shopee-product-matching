import timm
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveGeneralizedMeanPool2d(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(AdaptiveGeneralizedMeanPool2d, self).__init__()
        self.p = p
        self.eps = eps
        self.flatten1 = nn.Flatten()

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        x = F.adaptive_avg_pool2d(input=x.clamp(min=eps).pow(p), output_size=(1, 1)).pow(1.0 / p)
        x = self.flatten1(x)
        return x


class SimpleMobileNetV3(nn.Module):
    def __init__(self, embedding_size=256, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )
        self.gem_pool = nn.Sequential(AdaptiveGeneralizedMeanPool2d(p=3), nn.Flatten())
        self.fc = nn.Linear(576, embedding_size)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.gem_pool(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x
