# app/preprocess.py

from torchvision import transforms

# 1. 질환명 리스트 (v2.0 릴리즈 기준, 순서 맞추기!)
disease_names = [
    "비듬",
    "미세각질",
    "모낭홍반/농포",
    "모낭사이홍반",
    "피지과다",
    "탈모"
]

# 2. 모델 파일 경로 (질환명 리스트 순서와 1:1로 맞춰야 함)
model_paths = [
    "app/model_weight/biddem_compressed.pt",  # 비듬
    "app/model_weight/mise_compressed.pt",    # 미세각질
    "app/model_weight/mono_compressed.pt",    # 모낭홍반/농포
    "app/model_weight/mosa_compressed.pt",    # 모낭사이홍반
    "app/model_weight/pizi_compressed.pt",    # 피지과다
    "app/model_weight/talmo_compressed.pt"    # 탈모
]

# 3. 전처리 함수 리스트 (merge.ipynb와 동일하게 맞춤, 모두 동일하면 *6)
default_transform = transforms.Compose([
    transforms.Resize((380, 380), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
preprocess_funcs = [default_transform for _ in range(len(disease_names))]

# 만약 일부 질환만 별도 전처리가 필요하면 아래처럼 각각 작성
# preprocess_funcs = [
#     transform1,  # 비듬
#     transform2,  # 미세각질
#     ...
# ]
