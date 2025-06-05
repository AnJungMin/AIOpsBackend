# app/api/model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

def build_efficientnet_b0_classifier(num_classes=4):
    """
    EfficientNet-B0 기반 커스텀 단일 분류기 생성 함수.
    (각 질환 pt파일에 사용)
    """
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model

def load_model(model_path, device):
    """
    pt 파일 경로와 device를 받아서 모델을 로드하여 반환.
    """
    model = build_efficientnet_b0_classifier(num_classes=4)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
