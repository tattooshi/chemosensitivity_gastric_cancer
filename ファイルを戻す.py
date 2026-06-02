import torch
from pathlib import Path
import shutil

info_path = Path(
    "/home/tatsushi/デスクトップ/val-result_weight/EVA02_0602_0.9/"
    "eva02_large_patch14_0448.mim_in22k_ft_in22k/"
    "Vit_fixed_val_classes_models_0sens_positive/"
    "top5_ensemble_info.pth"
)

txt_path = info_path.with_name("topk_model_paths.txt")
backup_path = info_path.with_name("top5_ensemble_info_backup_before_edit.pth")

# バックアップ
shutil.copy2(info_path, backup_path)

# pth読み込み
data = torch.load(info_path, map_location="cpu", weights_only=False)

# txtから読み込み
with open(txt_path, "r", encoding="utf-8") as f:
    new_paths = [line.strip() for line in f if line.strip()]

print("読み込んだパス:")
for p in new_paths:
    print(p)

# 個数チェック
if len(new_paths) != len(data["topk_model_paths"]):
    raise ValueError(
        f"数が違います: txt={len(new_paths)}, pth={len(data['topk_model_paths'])}"
    )

# 置き換え
data["topk_model_paths"] = new_paths

# 保存
torch.save(data, info_path)

print("\n保存しました:")
print(info_path)
print("\nバックアップ:")
print(backup_path)