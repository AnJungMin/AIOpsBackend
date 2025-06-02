from torchvision import transforms

# 1. 질환명 리스트 (설명해주신 순서대로)
disease_names = [
    "비듬",
    "미세각질",
    "모낭홍반/농포",
    "모낭사이홍반",
    "피지과다",
    "탈모"
]

# 2. 모델 파일 경로 (Dockerfile의 wget 경로와 반드시 동일!)
model_paths = [
    "app/model_weight/EfficientNet_B3_BIDDEM_V3.1_2025_05_24_19_model.pt",
    "app/model_weight/EfficientNet_B3_MISE_V4_2025_05_26_18_model.pt",
    "app/model_weight/EfficientNet_B3_MONO_V4_2025_05_27_16_model.pt",
    "app/model_weight/EfficientNet_B3_MOSA_V5_TUNNING_2025_06_01_23_final_model.pt",
    "app/model_weight/EfficientNet_B3_PIZI_V4_2025_05_27_17_model.pt",
    "app/model_weight/EfficientNet_B3_TALMO_V5_TUNNING_2025_06_01_16_model.pt"
]

# 3. 전처리 함수 리스트 (질환별로 다르게 하고 싶으면 여러 개)
preprocess_funcs = [
    transforms.Compose([
        transforms.Resize((380, 380), interpolation=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
] * 6   # 모두 동일하다면 곱하기 6, 다르게 하려면 각각 작성!
