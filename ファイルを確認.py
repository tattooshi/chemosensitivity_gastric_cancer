import torch
from pathlib import Path

pth_path = Path(
    "./EVA02_model/top5_ensemble_info.pth"
)

# 自分で作成した信頼できるpthなので weights_only=False で読む
data = torch.load(pth_path, map_location="cpu", weights_only=False)

print("type:", type(data))

if isinstance(data, dict):
    print("\nkeys:")
    for k, v in data.items():
        print(f"{k}: {type(v)}")

    print("\n--- 内容の概要 ---")
    for k, v in data.items():
        print(f"\n[{k}]")
        if isinstance(v, list):
            print("list length:", len(v))
            for i, item in enumerate(v[:10]):
                print(f"  {i}: {item}")
        elif isinstance(v, dict):
            print("dict keys:", list(v.keys()))
        else:
            print(v)
else:
    print(data)