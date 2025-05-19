# fairface_model.py
import torch.nn as nn
from torchvision.models import resnet34

class FairFaceResnet34(nn.Module):
    def __init__(self, num_classes=18):
        super(FairFaceResnet34, self).__init__()
        self.convnet = resnet34(weights=None)  # niente pretrained, per compatibilità
        self.convnet.fc = nn.Linear(self.convnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.convnet(x)
