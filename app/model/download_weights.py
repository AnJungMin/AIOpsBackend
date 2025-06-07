# app/model/download_weights.py

import os
import requests

MODEL_INFOS = [
    (
        "biddem_B0_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v3.0/biddem_B0_compressed.pt"
    ),
    (
        "mise_B0_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v3.0/mise_B0_compressed.pt"
    ),
    (
        "mono_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/mono_compressed.pt"
    ),
    (
        "mosa_B0_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v3.0/mosa_B0_compressed.pt"
    ),
    (
        "pizi_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v2.0/pizi_compressed.pt"
    ),
    (
        "talmo_B0_compressed.pt",
        "https://github.com/AnJungMin/AIOpsBackend/releases/download/v3.0/talmo_B0_compressed.pt"
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
