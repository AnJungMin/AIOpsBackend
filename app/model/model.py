import os
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

def build_efficientnet_b0_classifier(num_classes=3):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, num_classes)
    )
    return model

def load_model(model_path, device):
    """
    pt/pth 파일 경로와 device를 받아서 EfficientNet-B0 기반 모델을 로드하여 반환.
    파일이 dict 형태라면 'model' 키 내부의 state_dict를 사용.
    """
    model = build_efficientnet_b0_classifier(num_classes=3)
    state_dict = torch.load(model_path, map_location=device)
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
