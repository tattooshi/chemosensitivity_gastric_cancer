#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指定フォルダー内の画像を top5 ensemble で推論し、
1) 各画像ファイルごとの推論結果
2) フォルダー全体の mean / median 集約結果
を表示するコード

前提:
- class=1 が 0_sens の確率
- label mapping:
    2_resis -> 0
    0_sens  -> 1
- top5 ensemble 情報は 0_sens positive 版を使用

入力:
- GUIで選択した画像が入ったフォルダー
  （下層も再帰的に探索）

出力:
- 画面表示
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import timm
from timm.data import resolve_data_config, create_transform


ENSEMBLE_INFO_PATH = Path(
    "/home/tatsushi/デスクトップ/感受性AI/model/top5_ensemble_info.pth"
)

FEATURE_MODEL_NAME = "vit_large_patch16_384.augreg_in21k_ft_in1k"
IMG_SIZE = 616
VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# モデル
# =========================================================
class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================================================
# 安全な torch.load
# =========================================================
def torch_load_safe(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


# =========================================================
# 画像一覧取得
# =========================================================
def collect_image_files(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"フォルダーが存在しません: {folder}")

    img_files = []
    for root, _, files in os.walk(folder):
        root_path = Path(root)
        for f in files:
            p = root_path / f
            if p.suffix.lower() in VALID_EXT:
                img_files.append(p)

    img_files = sorted(img_files)
    if len(img_files) == 0:
        raise ValueError(f"画像ファイルが見つかりません: {folder}")

    return img_files


# =========================================================
# ensemble 読み込み
# =========================================================
def load_ensemble_models(ensemble_info_path: Path, device):
    info = torch_load_safe(ensemble_info_path, map_location="cpu")

    topk_model_paths = info["topk_model_paths"]
    cutoff = float(info.get("ensemble_best_cutoff", 0.5))

    models = []
    for model_path in topk_model_paths:
        ckpt = torch_load_safe(model_path, map_location="cpu")
        in_dim = ckpt["input_dim"]

        model = MLPClassifier(in_dim=in_dim).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models, cutoff, info


# =========================================================
# 特徴抽出モデル
# =========================================================
def load_feature_extractor(device):
    feat_model = timm.create_model(FEATURE_MODEL_NAME, pretrained=True, num_classes=0)
    feat_model.to(device).eval()
    for p in feat_model.parameters():
        p.requires_grad = False

    if hasattr(feat_model, "set_input_size"):
        feat_model.set_input_size(img_size=IMG_SIZE)
    else:
        raise RuntimeError(
            "This model does not support set_input_size(). "
            "Please update timm or use another model."
        )

    data_cfg = resolve_data_config({}, model=feat_model)
    data_cfg["input_size"] = (3, IMG_SIZE, IMG_SIZE)
    data_cfg["crop_pct"] = 1.0
    transform = create_transform(**data_cfg)

    return feat_model, transform


# =========================================================
# 画像 -> 特徴
# =========================================================
@torch.no_grad()
def extract_cls_feature(model, x: torch.Tensor) -> torch.Tensor:
    feats = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    if feats.ndim == 3:
        return feats[:, 0]
    if feats.ndim == 4:
        return feats.mean(dim=(2, 3))
    return feats


# =========================================================
# 1画像推論
# 戻り値:
#   prob_0_sens, prob_2_resis
# =========================================================
@torch.no_grad()
def predict_one_image_ensemble(models, feat_model, transform, image_path: Path, device):
    img = Image.open(image_path).convert("RGB")
    x_img = transform(img).unsqueeze(0).to(device)  # [1, 3, 616, 616]
    feat_1d = extract_cls_feature(feat_model, x_img).squeeze(0).float().reshape(-1)

    expected_dim = models[0].net[0].in_features
    if feat_1d.numel() != expected_dim:
        raise ValueError(
            f"特徴次元不一致: {image_path}\n"
            f"  expected={expected_dim}, got={feat_1d.numel()}"
        )

    x = feat_1d.unsqueeze(0).to(device)   # [1, D]

    prob_list = []
    for model in models:
        out = model(x)
        proba = torch.softmax(out, dim=1)[:, 1]   # class=1 = 0_sens
        prob_list.append(proba.item())

    prob_0_sens = float(np.mean(prob_list))
    prob_2_resis = 1.0 - prob_0_sens

    return prob_0_sens, prob_2_resis


# =========================================================
# クラス文字列
# =========================================================
def prob_to_class(prob_0_sens: float, cutoff: float):
    pred_label = int(prob_0_sens >= cutoff)
    pred_class = "0_sens" if pred_label == 1 else "2_resis"
    return pred_label, pred_class


def select_target_folder() -> Path:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askdirectory(title="画像ファイルが入ったフォルダーを選択してください")
    root.destroy()

    if not selected:
        raise ValueError("フォルダーが選択されませんでした。処理を終了します。")

    return Path(selected)


# =========================================================
# main
# =========================================================
def main():
    target_folder = select_target_folder()

    print("===== load models =====")
    feat_model, transform = load_feature_extractor(DEVICE)
    models, cutoff, _ = load_ensemble_models(ENSEMBLE_INFO_PATH, DEVICE)
    input_dim = models[0].net[0].in_features

    print(f"device   : {DEVICE}")
    print(f"feature  : {FEATURE_MODEL_NAME}")
    print(f"img_size : {IMG_SIZE}x{IMG_SIZE}")
    print(f"n_models : {len(models)}")
    print(f"in_dim   : {input_dim}")
    print(f"cutoff   : {cutoff:.4f}")
    print(f"folder   : {target_folder}")

    print("\n===== collect image files =====")
    image_files = collect_image_files(target_folder)
    print(f"n_images : {len(image_files)}")

    prob_0_sens_list = []
    valid_count = 0

    print("\n===== per-image inference =====")
    for i, image_path in enumerate(image_files, start=1):
        try:
            prob_0_sens, prob_2_resis = predict_one_image_ensemble(
                models, feat_model, transform, image_path, DEVICE
            )
        except Exception as e:
            print(f"[{i:03d}/{len(image_files):03d}] {image_path.name} | SKIP | {e}")
            continue

        valid_count += 1
        prob_0_sens_list.append(prob_0_sens)
        _, pred_class = prob_to_class(prob_0_sens, cutoff)

        print(
            f"[{i:03d}/{len(image_files):03d}] "
            f"{image_path.name} | "
            f"pred={pred_class} | "
            f"p(0_sens)={prob_0_sens:.4f} | "
            f"p(2_resis)={prob_2_resis:.4f}"
        )

    if valid_count == 0:
        raise RuntimeError("有効な画像を1件も処理できませんでした。")

    # -------------------------------------------------
    # フォルダー全体集約
    # -------------------------------------------------
    mean_prob_0_sens = float(np.mean(prob_0_sens_list))
    median_prob_0_sens = float(np.median(prob_0_sens_list))

    mean_prob_2_resis = 1.0 - mean_prob_0_sens
    median_prob_2_resis = 1.0 - median_prob_0_sens

    mean_pred_label, mean_pred_class = prob_to_class(mean_prob_0_sens, cutoff)
    median_pred_label, median_pred_class = prob_to_class(median_prob_0_sens, cutoff)

    # -------------------------------------------------
    # 表示
    # -------------------------------------------------
    print("\n===== folder summary =====")
    print(f"cutoff = {cutoff:.4f}")
    print(f"n_valid_images = {valid_count}")
    print(f"mean_pred_label   = {mean_pred_label}")
    print(f"median_pred_label = {median_pred_label}")

    print("\n[MEAN]")
    print(f"pred_class   : {mean_pred_class}")
    print(f"prob_0_sens  : {mean_prob_0_sens:.4f}")
    print(f"prob_2_resis : {mean_prob_2_resis:.4f}")
    print(f"margin       : {mean_prob_0_sens - cutoff:+.4f}")

    print("\n[MEDIAN]")
    print(f"pred_class   : {median_pred_class}")
    print(f"prob_0_sens  : {median_prob_0_sens:.4f}")
    print(f"prob_2_resis : {median_prob_2_resis:.4f}")
    print(f"margin       : {median_prob_0_sens - cutoff:+.4f}")


if __name__ == "__main__":
    main()
