import torch
from pathlib import Path
import shutil

script_dir = Path(__file__).resolve().parent

info_path = script_dir / "model" / "top5_ensemble_info.pth"

txt_path = script_dir / "topk_model_paths.txt"
backup_path = script_dir / "top5_ensemble_info_backup_before_edit.pth"

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
