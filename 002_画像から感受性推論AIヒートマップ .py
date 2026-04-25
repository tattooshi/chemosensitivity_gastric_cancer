# 胃癌の化学療法感受性を推論するAI

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

変更点:
- 画像はまず「中央の最大正方形」を明示的に切り出す
- その後に 616x616 へ変換して特徴抽出する
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
from PIL import Image, ImageTk
import timm
from timm.data import resolve_data_config, create_transform

ENSEMBLE_INFO_PATH = Path(
    "./models/top5_ensemble_info.pth"
)

FEATURE_MODEL_NAME = "vit_large_patch16_384.augreg_in21k_ft_in1k"
IMG_SIZE = 616
VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DISPLAY_NAMES = {
    "0_sens": "chemo-sensitive",
    "2_resis": "chemo-resistance",
}


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
# 中央の最大正方形でトリミング
# =========================================================
def center_crop_max_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)

    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side

    return img.crop((left, top, right, bottom))


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

    cls_token = feats[:, 0:1, :]
    patch_tokens = feats[:, 1:, :]

    rel = F.cosine_similarity(
        patch_tokens,
        cls_token.expand_as(patch_tokens),
        dim=-1
    )

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


def display_class_name(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(class_name, class_name)


def format_display_text(text: str) -> str:
    text = text.replace("p(0_sens)", "p(chemo-sensitive)")
    text = text.replace("p(2_resis)", "p(chemo-resistance)")
    text = text.replace("prob_0_sens", "prob_chemo_sensitive")
    text = text.replace("prob_2_resis", "prob_chemo_resistance")
    for raw_name, display_name in CLASS_DISPLAY_NAMES.items():
        text = text.replace(raw_name, display_name)
    return text


def show_result_tk(
    image_path: Path,
    prob_0_sens: float,
    cutoff: float,
    orig_img: Image.Image,
    heat_img: Image.Image,
    overlay_img: Image.Image,
    terminal_output: str = ""
):
    root = tk.Tk()
    root.title("Chemosensitivity AI Result")
    root.configure(bg="#f4f7fb")
    root.minsize(1120, 760)

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("App.TFrame", background="#f4f7fb")
    style.configure("Card.TFrame", background="#ffffff", relief="flat")
    style.configure("Header.TLabel", background="#f4f7fb", foreground="#111827", font=("Arial", 18, "bold"))
    style.configure("Subtle.TLabel", background="#f4f7fb", foreground="#64748b", font=("Arial", 10))
    style.configure("CardTitle.TLabel", background="#ffffff", foreground="#334155", font=("Arial", 10, "bold"))
    style.configure("Metric.TLabel", background="#ffffff", foreground="#0f172a", font=("Arial", 11))
    style.configure("ImageTitle.TLabel", background="#ffffff", foreground="#334155", font=("Arial", 11, "bold"))
    style.configure("Log.TLabelframe", background="#f4f7fb", foreground="#334155")
    style.configure("Log.TLabelframe.Label", background="#f4f7fb", foreground="#334155", font=("Arial", 10, "bold"))

    pred_label, pred_class = prob_to_class(prob_0_sens, cutoff)
    pred_display = display_class_name(pred_class)
    prob_2_resis = 1.0 - prob_0_sens
    accent_color = "#0f9f6e" if pred_class == "0_sens" else "#dc2626"
    accent_bg = "#d1fae5" if pred_class == "0_sens" else "#fee2e2"

    app = ttk.Frame(root, style="App.TFrame", padding=(18, 16))
    app.pack(fill="both", expand=True)

    ttk.Label(app, text="Chemosensitivity AI Result", style="Header.TLabel").pack(anchor="w")
    ttk.Label(app, text=image_path.name, style="Subtle.TLabel").pack(anchor="w", pady=(2, 14))

    summary = ttk.Frame(app, style="Card.TFrame", padding=16)
    summary.pack(fill="x", pady=(0, 14))

    pred_badge = tk.Label(
        summary,
        text=pred_display,
        bg=accent_bg,
        fg=accent_color,
        font=("Arial", 18, "bold"),
        padx=18,
        pady=10,
        relief="flat"
    )
    pred_badge.grid(row=0, column=0, rowspan=2, sticky="nsw", padx=(0, 18))

    metrics = [
        ("Label", str(pred_label)),
        ("p(chemo-sensitive)", f"{prob_0_sens:.4f}"),
        ("p(chemo-resistance)", f"{prob_2_resis:.4f}"),
        ("Cutoff", f"{cutoff:.4f}"),
        ("Margin", f"{prob_0_sens - cutoff:+.4f}"),
    ]
    for idx, (label, value) in enumerate(metrics):
        col = idx + 1
        ttk.Label(summary, text=label, style="CardTitle.TLabel").grid(row=0, column=col, sticky="w", padx=8)
        ttk.Label(summary, text=value, style="Metric.TLabel").grid(row=1, column=col, sticky="w", padx=8, pady=(4, 0))

    frame = ttk.Frame(app, style="App.TFrame")
    frame.pack(fill="both", expand=True, pady=(0, 14))

    def fit_to_max_height(img: Image.Image, max_h: int = 320):
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

    root._img_refs = [tk_orig, tk_heat, tk_over]

    for col, (title, image_ref) in enumerate([
        ("Original", tk_orig),
        ("Heatmap", tk_heat),
        ("Overlay", tk_over),
    ]):
        panel = ttk.Frame(frame, style="Card.TFrame", padding=10)
        panel.grid(row=0, column=col, sticky="nsew", padx=6)
        frame.columnconfigure(col, weight=1)
        ttk.Label(panel, text=title, style="ImageTitle.TLabel").pack(anchor="w", pady=(0, 8))
        tk.Label(panel, image=image_ref, bg="#ffffff", bd=0).pack()

    log_frame = ttk.LabelFrame(app, text="Terminal Output", style="Log.TLabelframe", padding=8)
    log_frame.pack(fill="both", expand=True)

    log_text = scrolledtext.ScrolledText(
        log_frame,
        wrap="none",
        height=14,
        font=("Courier New", 10),
        bg="#0f172a",
        fg="#dbeafe",
        insertbackground="#dbeafe",
        selectbackground="#2563eb",
        selectforeground="#ffffff",
        relief="flat",
        borderwidth=0
    )
    log_text.pack(fill="both", expand=True)
    log_text.insert("1.0", format_display_text(terminal_output))
    log_text.configure(state="disabled")

    root.mainloop()


def make_dual_printer():
    logs = []

    def log_print(*args, sep=" ", end="\n"):
        text = sep.join(str(a) for a in args)
        print(text, end=end)
        logs.append(text + end)

    return log_print, logs

# =========================================================
# 1画像推論
# 戻り値:
#   prob_0_sens, prob_2_resis, feat_1d(np.ndarray)
# =========================================================
@torch.no_grad()
def predict_one_image_ensemble(models, feat_model, transform, image_path: Path, device):
    img = Image.open(image_path).convert("RGB")
    img = center_crop_max_square(img)   # ← 中央最大正方形を明示的に切る
    x_img = transform(img).unsqueeze(0).to(device)  # [1, 3, 616, 616]

    feat_1d = extract_cls_feature(feat_model, x_img).squeeze(0).float().reshape(-1)

    expected_dim = models[0].net[0].in_features
    if feat_1d.numel() != expected_dim:
        raise ValueError(
            f"特徴次元不一致: {image_path}\n"
            f"  expected={expected_dim}, got={feat_1d.numel()}"
        )

    x = feat_1d.unsqueeze(0).to(device)

    prob_list = []
    for model in models:
        out = model(x)
        proba = torch.softmax(out, dim=1)[:, 1]
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
    log_print, terminal_logs = make_dual_printer()
    target_folder = select_target_folder()

    log_print("===== load models =====")
    feat_model, transform = load_feature_extractor(DEVICE)
    models, cutoff, _ = load_ensemble_models(ENSEMBLE_INFO_PATH, DEVICE)
    input_dim = models[0].net[0].in_features

    log_print(f"device   : {DEVICE}")
    log_print(f"feature  : {FEATURE_MODEL_NAME}")
    log_print(f"img_size : {IMG_SIZE}x{IMG_SIZE}")
    log_print(f"n_models : {len(models)}")
    log_print(f"in_dim   : {input_dim}")
    log_print(f"cutoff   : {cutoff:.4f}")
    log_print(f"folder   : {target_folder}")

    log_print("\n===== collect image files =====")
    image_files = collect_image_files(target_folder)
    log_print(f"n_images : {len(image_files)}")

    prob_0_sens_list = []
    feature_list = []
    valid_image_paths = []
    valid_count = 0

    log_print("\n===== per-image inference =====")
    for i, image_path in enumerate(image_files, start=1):
        try:
            prob_0_sens, prob_2_resis, feat_vec = predict_one_image_ensemble(
                models, feat_model, transform, image_path, DEVICE
            )
        except Exception as e:
            log_print(f"[{i:03d}/{len(image_files):03d}] {image_path.name} | SKIP | {e}")
            continue

        valid_count += 1
        prob_0_sens_list.append(prob_0_sens)
        feature_list.append(feat_vec)
        valid_image_paths.append(image_path)
        _, pred_class = prob_to_class(prob_0_sens, cutoff)

        log_print(
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

    log_print("\n===== folder summary =====")
    log_print(f"cutoff = {cutoff:.4f}")
    log_print(f"n_valid_images = {valid_count}")
    log_print(f"mean_pred_label   = {mean_pred_label}")
    log_print(f"median_pred_label = {median_pred_label}")

    log_print("\n[MEAN]")
    log_print(f"pred_class   : {mean_pred_class}")
    log_print(f"prob_0_sens  : {mean_prob_0_sens:.4f}")
    log_print(f"prob_2_resis : {mean_prob_2_resis:.4f}")
    log_print(f"margin       : {mean_prob_0_sens - cutoff:+.4f}")

    log_print("\n[MEDIAN]")
    log_print(f"pred_class   : {median_pred_class}")
    log_print(f"prob_0_sens  : {median_prob_0_sens:.4f}")
    log_print(f"prob_2_resis : {median_prob_2_resis:.4f}")
    log_print(f"margin       : {median_prob_0_sens - cutoff:+.4f}")

    # -------------------------------------------------
    # 特徴ベクトル中央値に最も近い画像を可視化
    # -------------------------------------------------
    feats = np.stack(feature_list, axis=0)
    med_vec = np.median(feats, axis=0)
    dists = np.linalg.norm(feats - med_vec[None, :], axis=1)
    med_idx = int(np.argmin(dists))

    median_image_path = valid_image_paths[med_idx]
    median_image_prob = float(prob_0_sens_list[med_idx])

    log_print("\n===== median-feature image =====")
    log_print(f"selected_image : {median_image_path}")
    log_print(f"distance_to_feature_median : {dists[med_idx]:.6f}")
    log_print(f"p(0_sens) of selected image: {median_image_prob:.4f}")

    orig_img = Image.open(median_image_path).convert("RGB")
    orig_img = center_crop_max_square(orig_img)   # ← 推論時と同じ前処理
    x_img = transform(orig_img).unsqueeze(0).to(DEVICE)

    rel_map = extract_vit_token_relevance_map(feat_model, x_img)
    heat_img, overlay_img = make_heatmap_and_overlay(orig_img, rel_map, alpha=0.45)

    terminal_output = "".join(terminal_logs)

    show_result_tk(
        image_path=median_image_path,
        prob_0_sens=median_image_prob,
        cutoff=cutoff,
        orig_img=orig_img,
        heat_img=heat_img,
        overlay_img=overlay_img,
        terminal_output=terminal_output,
    )


if __name__ == "__main__":
    main()
