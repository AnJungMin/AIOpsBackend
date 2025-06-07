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
    "app/model_weight/mono_compressed.pt",        # 모낭홍반/농포(B3)
    "app/model_weight/mosa_B0_compressed.pt",     # 모낭사이홍반(B0)
    "app/model_weight/pizi_B0_compressed.pt",        # 피지과다(B3)
    "app/model_weight/talmo_B0_compressed.pt"     # 탈모(B0)
]

def get_transform(model_path):
    size = (224, 224) if "_b0_" in model_path.lower() else (300, 300)
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

preprocess_funcs = [get_transform(path) for path in model_paths]
