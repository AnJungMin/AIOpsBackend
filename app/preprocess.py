from torchvision import transforms

# 1. 질환명 리스트 (v2.0 릴리즈 기준, 순서 중요)
disease_names = [
    "비듬",
    "미세각질",
    "모낭홍반/농포",
    "모낭사이홍반",
    "피지과다",
    "탈모"
]

# 2. 모델 파일 경로 (v2.0 릴리즈 파일명 기준, 순서 맞춰야 함)
model_paths = [
    "app/model_weight/biddem_compressed.pt",  # 비듬
    "app/model_weight/mise_compressed.pt",    # 미세각질
    "app/model_weight/mono_compressed.pt",    # 모낭홍반/농포
    "app/model_weight/mosa_compressed.pt",    # 모낭사이홍반
    "app/model_weight/pizi_compressed.pt",    # 피지과다
    "app/model_weight/talmo_compressed.pt"    # 탈모
]

# 3. 전처리 함수 리스트 (merge.ipynb와 동일하게 380x380, mean/std 값 확인)
preprocess_funcs = [
    transforms.Compose([
        transforms.Resize((380, 380), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
] * 6   # 모두 동일하다면 곱하기 6, 다르게 하려면 각각 작성!
