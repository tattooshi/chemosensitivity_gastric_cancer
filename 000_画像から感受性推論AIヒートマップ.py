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
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
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
# ViT token relevance map (Grad-CAM風)
# =========================================================
@torch.no_grad()
def extract_vit_token_relevance_map(model, x: torch.Tensor) -> np.ndarray:
    feats = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
    if feats.ndim != 3 or feats.shape[1] < 2:
        raise RuntimeError("ViTのtoken表現が取得できませんでした。")

    cls_token = feats[:, 0:1, :]      # [1,1,C]
    patch_tokens = feats[:, 1:, :]    # [1,P,C]

    # CLSと各patch tokenのcos類似度を「寄与っぽさ」として可視化する
    rel = F.cosine_similarity(patch_tokens, cls_token.expand_as(patch_tokens), dim=-1)  # [1,P]

    num_patches = rel.shape[1]
    if hasattr(model, "patch_embed") and hasattr(model.patch_embed, "grid_size"):
        grid_h, grid_w = model.patch_embed.grid_size
        if grid_h * grid_w != num_patches:
            side = int(np.sqrt(num_patches))
            grid_h, grid_w = side, side
    else:
        side = int(np.sqrt(num_patches))
        grid_h, grid_w = side, side

    rel_map = rel.view(1, 1, grid_h, grid_w)
    rel_map = F.interpolate(
        rel_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
    )
    rel_map = rel_map.squeeze().float().cpu().numpy()
    rel_map = rel_map - rel_map.min()
    rel_map = rel_map / (rel_map.max() + 1e-8)
    return rel_map


# =========================================================
# 可視化ユーティリティ
# =========================================================
def jet_colormap(gray: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * g - 3.0), 0.0, 1.0)
    gg = np.clip(1.5 - np.abs(4.0 * g - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * g - 1.0), 0.0, 1.0)
    return np.stack([r, gg, b], axis=-1)


def make_heatmap_and_overlay(orig_img: Image.Image, rel_map: np.ndarray, alpha: float = 0.45):
    w, h = orig_img.size
    rel_img = Image.fromarray((rel_map * 255).astype(np.uint8), mode="L").resize((w, h), Image.BILINEAR)
    rel = np.array(rel_img).astype(np.float32) / 255.0

    heat_rgb = (jet_colormap(rel) * 255.0).astype(np.uint8)
    orig_rgb = np.array(orig_img.convert("RGB")).astype(np.float32)
    overlay = np.clip((1.0 - alpha) * orig_rgb + alpha * heat_rgb.astype(np.float32), 0, 255).astype(np.uint8)

    heat_pil = Image.fromarray(heat_rgb, mode="RGB")
    overlay_pil = Image.fromarray(overlay, mode="RGB")
    return heat_pil, overlay_pil


def show_result_tk(image_path: Path, prob_0_sens: float, cutoff: float, orig_img: Image.Image, heat_img: Image.Image, overlay_img: Image.Image):
    root = tk.Tk()
    root.title("Median Feature Image + Heatmap")

    pred_label, pred_class = prob_to_class(prob_0_sens, cutoff)
    info = (
        f"image: {image_path.name}\n"
        f"pred: {pred_class} (label={pred_label})  "
        f"p(0_sens)={prob_0_sens:.4f}  cutoff={cutoff:.4f}"
    )
    tk.Label(root, text=info, justify="left", anchor="w", font=("Arial", 11)).pack(fill="x", padx=8, pady=8)

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True, padx=8, pady=8)

    def fit_to_max_height(img: Image.Image, max_h: int = 480):
        w, h = img.size
        if h <= max_h:
            return img
        scale = max_h / float(h)
        return img.resize((int(w * scale), max_h), Image.BILINEAR)

    orig_v = fit_to_max_height(orig_img)
    heat_v = fit_to_max_height(heat_img)
    over_v = fit_to_max_height(overlay_img)

    tk_orig = ImageTk.PhotoImage(orig_v)
    tk_heat = ImageTk.PhotoImage(heat_v)
    tk_over = ImageTk.PhotoImage(over_v)

    root._img_refs = [tk_orig, tk_heat, tk_over]  # gc対策

    col1 = tk.Frame(frame)
    col1.pack(side="left", padx=6)
    tk.Label(col1, text="Original").pack()
    tk.Label(col1, image=tk_orig).pack()

    col2 = tk.Frame(frame)
    col2.pack(side="left", padx=6)
    tk.Label(col2, text="Heatmap").pack()
    tk.Label(col2, image=tk_heat).pack()

    col3 = tk.Frame(frame)
    col3.pack(side="left", padx=6)
    tk.Label(col3, text="Overlay").pack()
    tk.Label(col3, image=tk_over).pack()

    root.mainloop()


# =========================================================
# 1画像推論
# 戻り値:
#   prob_0_sens, prob_2_resis, feat_1d(np.ndarray)
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

    return prob_0_sens, prob_2_resis, feat_1d.detach().cpu().numpy()


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
    feature_list = []
    valid_image_paths = []
    valid_count = 0

    print("\n===== per-image inference =====")
    for i, image_path in enumerate(image_files, start=1):
        try:
            prob_0_sens, prob_2_resis, feat_vec = predict_one_image_ensemble(
                models, feat_model, transform, image_path, DEVICE
            )
        except Exception as e:
            print(f"[{i:03d}/{len(image_files):03d}] {image_path.name} | SKIP | {e}")
            continue

        valid_count += 1
        prob_0_sens_list.append(prob_0_sens)
        feature_list.append(feat_vec)
        valid_image_paths.append(image_path)
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

    # -------------------------------------------------
    # 特徴ベクトル中央値に最も近い画像を可視化
    # -------------------------------------------------
    feats = np.stack(feature_list, axis=0)  # [N, D]
    med_vec = np.median(feats, axis=0)
    dists = np.linalg.norm(feats - med_vec[None, :], axis=1)
    med_idx = int(np.argmin(dists))

    median_image_path = valid_image_paths[med_idx]
    median_image_prob = float(prob_0_sens_list[med_idx])

    print("\n===== median-feature image =====")
    print(f"selected_image : {median_image_path}")
    print(f"distance_to_feature_median : {dists[med_idx]:.6f}")
    print(f"p(0_sens) of selected image: {median_image_prob:.4f}")

    orig_img = Image.open(median_image_path).convert("RGB")
    x_img = transform(orig_img).unsqueeze(0).to(DEVICE)
    rel_map = extract_vit_token_relevance_map(feat_model, x_img)
    heat_img, overlay_img = make_heatmap_and_overlay(orig_img, rel_map, alpha=0.45)

    show_result_tk(
        image_path=median_image_path,
        prob_0_sens=median_image_prob,
        cutoff=cutoff,
        orig_img=orig_img,
        heat_img=heat_img,
        overlay_img=overlay_img,
    )


if __name__ == "__main__":
    main()
