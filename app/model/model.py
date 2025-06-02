# app/api/model.py
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b3

def build_efficientnet_b3_classifier(num_classes=3):
    """
    EfficientNet-B3 기반 커스텀 단일 분류기 생성 함수.
    (각 질환 pt파일에 사용)
    """
    model = efficientnet_b3()
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
    pt 파일 경로와 device를 받아서 모델을 로드하여 반환.
    """
    model = build_efficientnet_b3_classifier(num_classes=3)
    # pt파일 저장 시 {"model": model.state_dict()}로 저장된 경우
    state_dict = torch.load(model_path, map_location=device)['model']
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model
