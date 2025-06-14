# app/preprocess.py

from torchvision import transforms

# 1. 질환명 리스트 (순서 주의)
disease_names = [
    "비듬",
    "미세각질",
    "모낭홍반/농포",
    "모낭사이홍반",
    "피지과다",
    "탈모"
]

# 2. 모델 파일 경로 (질환명과 순서 1:1로 일치해야 함)
model_paths = [
    "app/model_weight/biddem_B0_compressed.pt",  # 비듬
    "app/model_weight/mise_B0_compressed.pt",    # 미세각질
    "app/model_weight/mono_B0_compressed.pt",    # 모낭홍반/농포
    "app/model_weight/mosa_B0_compressed.pt",    # 모낭사이홍반
    "app/model_weight/pizi_B0_compressed.pt",    # 피지과다
    "app/model_weight/talmo_B0_compressed.pt"    # 탈모
]

# 3. 전처리 함수 (merge.ipynb와 동일, 모두 동일하면 *6 활용)
default_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
preprocess_funcs = [default_transform] * len(disease_names)  # 리스트 길이 동적 계산
