# models/resnet.py

import torch.nn as nn
from torchvision.models import resnet34 as tv_resnet34

def resnet34(num_classes=18):
    model = tv_resnet34(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
