import torch
from pathlib import Path

info_path = Path(
    "/home/tatsushi/デスクトップ/val-result_weight/EVA02_0602_0.9/"
    "eva02_large_patch14_0448.mim_in22k_ft_in22k/"
    "Vit_fixed_val_classes_models_0sens_positive/"
    "top5_ensemble_info.pth"
)

txt_path = info_path.with_name("topk_model_paths.txt")

data = torch.load(info_path, map_location="cpu", weights_only=False)

with open(txt_path, "w", encoding="utf-8") as f:
    for p in data["topk_model_paths"]:
        f.write(str(p) + "\n")

print("書き出しました:")
print(txt_path)