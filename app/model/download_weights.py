# app/model/download_weights.py

import os
import requests

MODEL_INFOS = [
    (
        "EfficientNet_B3_BIDDEM_V3.1_2025_05_24_19_model.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_BIDDEM_V3.1_2025_05_24_19_model.pt"
    ),
    (
        "EfficientNet_B3_MISE_V4_2025_05_26_18_model.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_MISE_V4_2025_05_26_18_model.pt"
    ),
    (
        "EfficientNet_B3_MONO_V4_2025_05_27_16_model.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_MONO_V4_2025_05_27_16_model.pt"
    ),
    (
        "EfficientNet_B3_MOSA_V5_TUNNING_2025_06_01_23_final_model.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_MOSA_V5_TUNNING_2025_06_01_23_final_model.pt"
    ),
    (
        "EfficientNet_B3_PIZI_V4_2025_05_27_17_model.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_PIZI_V4_2025_05_27_17_model.pt"
    ),
    (
        "EfficientNet_B3_TALMO_V5_TUNNING_2025_06_01_16_model.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v1.0/EfficientNet_B3_TALMO_V5_TUNNING_2025_06_01_16_model.pt"
    ),
]

os.makedirs("app/model_weight", exist_ok=True)

for fname, url in MODEL_INFOS:
    fpath = os.path.join("app/model_weight", fname)
    if not os.path.exists(fpath):
        print(f"Downloading: {fname}")
        r = requests.get(url)
        if r.status_code == 200:
            with open(fpath, "wb") as f:
                f.write(r.content)
            print(f"Saved: {fpath}")
        else:
            print(f"Failed to download {fname} ({url}) - Status code: {r.status_code}")
    else:
        print(f"Already exists: {fname}")
