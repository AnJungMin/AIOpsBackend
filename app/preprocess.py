from torchvision import transforms

disease_names = [
    "비듬",
    "미세각질",
    "모낭홍반/농포",
    "모낭사이홍반",
    "피지과다",
    "탈모"
]

model_paths = [
    "app/model_weight/biddem_B0_compressed.pt",   # 비듬(B0)
    "app/model_weight/mise_B0_compressed.pt",     # 미세각질(B0)
    "app/model_weight/mono_B0_compressed.pt",     # 모낭홍반/농포(B0)
    "app/model_weight/mosa_B0_compressed.pt",     # 모낭사이홍반(B0)
    "app/model_weight/pizi_B0_compressed.pt",     # 피지과다(B0)
    "app/model_weight/talmo_B0_compressed.pt"     # 탈모(B0)
]

# 모든 모델에 동일한 전처리 적용
default_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
preprocess_funcs = [default_transform] * len(disease_names)
