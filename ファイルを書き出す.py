import torch
from pathlib import Path

info_path = Path(
    "./model/top5_ensemble_info.pth"
)

txt_path = Path(__file__).resolve().parent / "topk_model_paths.txt"

data = torch.load(info_path, map_location="cpu", weights_only=False)

with open(txt_path, "w", encoding="utf-8") as f:
    for p in data["topk_model_paths"]:
        f.write(str(p) + "\n")

print("書き出しました:")
print(txt_path)
