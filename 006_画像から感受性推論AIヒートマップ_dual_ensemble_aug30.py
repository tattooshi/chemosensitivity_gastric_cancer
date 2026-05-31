# 胃癌の化学療法感受性を ViT-CLS + EVA02-AVG でensemble推論し、medianで最終判定するAI

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
指定フォルダー内の画像を、既存の2系統モデルで推論してensembleするコード。
最終判定にはフォルダー内画像の median logit margin を使う。

使用モデル:
- 002系: ViT-Large / 616px / CLS特徴 / top5 ensemble
- 003系: EVA02-Large / 448px / AVG patch token特徴 / top5 ensemble

既存の 002 / 003 の .py は変更しない。

最終判定:
- 各モデルの p(0_sens) と cutoff を logit に変換し、logit margin を計算する
- 画像ごとに2モデルの logit margin を重み付き平均する
- フォルダー全体では画像ごとの combined logit margin の median を最終判定に使う
- 片方の p(0_sens) がほぼ0%の場合は、もう片方が十分高信頼でない限り 2_resis 側に倒す
- 表示用の combined p(0_sens) は2モデルの確率の重み付き平均
"""

import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, scrolledtext
from PIL import Image, ImageTk
import timm
from timm.data import resolve_data_config, create_transform


VALID_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2モデルを同じ重みで合成する。検証データで最適化する場合はここを調整する。
VIT_WEIGHT = 0.5
EVA02_WEIGHT = 0.5

# 各モデルの cutoff が大きく異なる場合、単純な p-cutoff では「ほぼ0%」の強い陰性証拠が
# 弱く扱われることがあるため、判定用には cutoff を基準にした log-odds margin を使う。
PROB_EPS = 1e-6

# 片方がほぼ0_sensなしと判断した場合の安全弁。
# もう片方が十分高信頼で0_sensを示す場合だけ veto を解除する。
LOW_CONFIDENCE_SENS_VETO_PROB = 0.05
VETO_OVERRIDE_SENS_PROB = 0.85


@dataclass(frozen=True)
class BackendSpec:
    name: str
    ensemble_info_path: Path
    feature_model_name: str
    img_size: int
    pooling: str
    weight: float


VIT_SPEC = BackendSpec(
    name="ViT-CLS",
    ensemble_info_path=Path(
        "/home/tatsushi/デスクトップ/val-result_weight/ViT616-100-weight1/vit_large_patch16_616.augreg_in21k_ft_in1k/Vit_fixed_val_classes_models_0sens_positive/top5_ensemble_info.pth"
    ),
    feature_model_name="vit_large_patch16_384.augreg_in21k_ft_in1k",
    img_size=616,
    pooling="cls",
    weight=VIT_WEIGHT,
)

EVA02_SPEC = BackendSpec(
    name="EVA02-AVG",
    ensemble_info_path=Path(
        "/home/tatsushi/デスクトップ/val-result_weight/EVA02-100-weight1/eva02_large_patch14_0448.mim_in22k_ft_in22k/Vit_fixed_val_classes_models_0sens_positive/top5_ensemble_info.pth"
    ),
    feature_model_name="eva02_large_patch14_448.mim_in22k_ft_in22k",
    img_size=448,
    pooling="avg_patch_tokens",
    weight=EVA02_WEIGHT,
)


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def torch_load_safe(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=False)


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


def center_crop_max_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def load_ensemble_models(ensemble_info_path: Path, device):
    info = torch_load_safe(ensemble_info_path, map_location="cpu")
    topk_model_paths = info["topk_model_paths"]
    cutoff = float(info.get("ensemble_best_cutoff", 0.5))

    models = []
    for model_path in topk_model_paths:
        ckpt = torch_load_safe(model_path, map_location="cpu")
        model = MLPClassifier(in_dim=ckpt["input_dim"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

    return models, cutoff, info


def load_feature_extractor(spec: BackendSpec, device):
    feat_model = timm.create_model(spec.feature_model_name, pretrained=True, num_classes=0)
    feat_model.to(device).eval()
    for p in feat_model.parameters():
        p.requires_grad = False

    if hasattr(feat_model, "set_input_size"):
        feat_model.set_input_size(img_size=spec.img_size)
    else:
        raise RuntimeError(
            f"{spec.name}: This model does not support set_input_size(). "
            "Please update timm or use another model."
        )

    data_cfg = resolve_data_config({}, model=feat_model)
    data_cfg["input_size"] = (3, spec.img_size, spec.img_size)
    data_cfg["crop_pct"] = 1.0
    transform = create_transform(**data_cfg)
    return feat_model, transform


def load_backend(spec: BackendSpec, device):
    feat_model, transform = load_feature_extractor(spec, device)
    models, cutoff, info = load_ensemble_models(spec.ensemble_info_path, device)
    return {
        "spec": spec,
        "feat_model": feat_model,
        "transform": transform,
        "models": models,
        "cutoff": cutoff,
        "info": info,
    }


def logit(p: float) -> float:
    p = float(np.clip(p, PROB_EPS, 1.0 - PROB_EPS))
    return float(np.log(p / (1.0 - p)))


@torch.no_grad()
def extract_feature(model, x: torch.Tensor, pooling: str) -> torch.Tensor:
    feats = model.forward_features(x) if hasattr(model, "forward_features") else model(x)

    if feats.ndim == 3:
        if pooling == "cls":
            return feats[:, 0]
        if pooling == "avg_patch_tokens":
            if feats.shape[1] > 1:
                return feats[:, 1:].mean(dim=1)
            return feats.mean(dim=1)
        raise ValueError(f"未知のpoolingです: {pooling}")

    if feats.ndim == 4:
        return feats.mean(dim=(2, 3))
    return feats


@torch.no_grad()
def predict_with_backend(backend, cropped_img: Image.Image):
    spec = backend["spec"]
    x_img = backend["transform"](cropped_img).unsqueeze(0).to(DEVICE)
    feat_1d = (
        extract_feature(backend["feat_model"], x_img, spec.pooling)
        .squeeze(0)
        .float()
        .reshape(-1)
    )

    expected_dim = backend["models"][0].net[0].in_features
    if feat_1d.numel() != expected_dim:
        raise ValueError(
            f"{spec.name}: 特徴次元不一致 expected={expected_dim}, got={feat_1d.numel()}"
        )

    x = feat_1d.unsqueeze(0).to(DEVICE)
    prob_list = []
    for model in backend["models"]:
        out = model(x)
        proba = torch.softmax(out, dim=1)[:, 1]
        prob_list.append(proba.item())

    prob_0_sens = float(np.mean(prob_list))
    margin = logit(prob_0_sens) - logit(float(backend["cutoff"]))
    return prob_0_sens, margin, feat_1d.detach().cpu().numpy()


def combine_predictions(vit_prob, vit_margin, eva_prob, eva_margin):
    total_weight = VIT_SPEC.weight + EVA02_SPEC.weight
    if total_weight <= 0:
        raise ValueError("ensemble weight の合計が0以下です。")

    combined_prob = (
        VIT_SPEC.weight * vit_prob + EVA02_SPEC.weight * eva_prob
    ) / total_weight
    combined_margin = (
        VIT_SPEC.weight * vit_margin + EVA02_SPEC.weight * eva_margin
    ) / total_weight
    low_sens_veto = min(vit_prob, eva_prob) <= LOW_CONFIDENCE_SENS_VETO_PROB
    high_sens_override = max(vit_prob, eva_prob) >= VETO_OVERRIDE_SENS_PROB
    if low_sens_veto and not high_sens_override:
        combined_margin = min(combined_margin, -PROB_EPS)
    return float(combined_prob), float(combined_margin)


def prob_margin_to_class(combined_margin: float):
    pred_label = int(combined_margin >= 0.0)
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


def make_dual_printer():
    logs = []

    def log_print(*args, sep=" ", end="\n"):
        text = sep.join(str(a) for a in args)
        print(text, end=end)
        logs.append(text + end)

    return log_print, logs


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    g = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * g - 3.0), 0.0, 1.0)
    gg = np.clip(1.5 - np.abs(4.0 * g - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * g - 1.0), 0.0, 1.0)
    return np.stack([r, gg, b], axis=-1)


@torch.no_grad()
def extract_token_relevance_map(backend, cropped_img: Image.Image) -> np.ndarray:
    spec = backend["spec"]
    x_img = backend["transform"](cropped_img).unsqueeze(0).to(DEVICE)
    feats = backend["feat_model"].forward_features(x_img)
    if feats.ndim != 3 or feats.shape[1] < 2:
        raise RuntimeError(f"{spec.name}: ViTのtoken表現が取得できませんでした。")

    patch_tokens = feats[:, 1:, :]
    if spec.pooling == "cls":
        ref_token = feats[:, 0:1, :]
    else:
        ref_token = patch_tokens.mean(dim=1, keepdim=True)

    rel = F.cosine_similarity(patch_tokens, ref_token.expand_as(patch_tokens), dim=-1)
    num_patches = rel.shape[1]

    if hasattr(backend["feat_model"], "patch_embed") and hasattr(
        backend["feat_model"].patch_embed, "grid_size"
    ):
        grid_h, grid_w = backend["feat_model"].patch_embed.grid_size
        if grid_h * grid_w != num_patches:
            side = int(np.sqrt(num_patches))
            grid_h, grid_w = side, side
    else:
        side = int(np.sqrt(num_patches))
        grid_h, grid_w = side, side

    rel_map = rel.view(1, 1, grid_h, grid_w)
    rel_map = F.interpolate(
        rel_map, size=(spec.img_size, spec.img_size), mode="bilinear", align_corners=False
    )
    rel_map = rel_map.squeeze().float().cpu().numpy()
    rel_map = rel_map - rel_map.min()
    rel_map = rel_map / (rel_map.max() + 1e-8)
    return rel_map


def make_heatmap_and_overlay(orig_img: Image.Image, rel_map: np.ndarray, alpha: float = 0.45):
    w, h = orig_img.size
    rel_img = Image.fromarray((rel_map * 255).astype(np.uint8), mode="L").resize(
        (w, h), Image.BILINEAR
    )
    rel = np.array(rel_img).astype(np.float32) / 255.0

    heat_rgb = (jet_colormap(rel) * 255.0).astype(np.uint8)
    orig_rgb = np.array(orig_img.convert("RGB")).astype(np.float32)
    overlay = np.clip(
        (1.0 - alpha) * orig_rgb + alpha * heat_rgb.astype(np.float32), 0, 255
    ).astype(np.uint8)
    return Image.fromarray(heat_rgb, mode="RGB"), Image.fromarray(overlay, mode="RGB")


def resize_for_display(img: Image.Image, max_h: int = 330) -> Image.Image:
    w, h = img.size
    if h <= max_h:
        return img
    scale = max_h / h
    return img.resize((int(w * scale), max_h), Image.BILINEAR)


def probability_color(prob: float) -> str:
    if prob >= 0.70:
        return "#16a34a"
    if prob >= 0.55:
        return "#86efac"
    if prob >= 0.45:
        return "#ffffbf"
    if prob >= 0.30:
        return "#fca5a5"
    return "#dc2626"


def prediction_color(pred_class: str) -> str:
    return "#16a34a" if pred_class == "0_sens" else "#dc2626"


def add_score_card(parent, title: str, value_text: str, prob: float, margin_text: str):
    card = tk.Frame(parent, bg="#111827", highlightbackground="#374151", highlightthickness=1)
    card.pack(side="left", fill="both", expand=True, padx=5)

    tk.Label(
        card,
        text=title,
        bg="#111827",
        fg="#e5e7eb",
        font=("Arial", 10, "bold"),
        anchor="w",
    ).pack(fill="x", padx=10, pady=(8, 2))

    tk.Label(
        card,
        text=value_text,
        bg="#111827",
        fg=probability_color(prob),
        font=("Arial", 18, "bold"),
        anchor="w",
    ).pack(fill="x", padx=10)

    bar_bg = tk.Canvas(card, height=14, bg="#1f2937", highlightthickness=0)
    bar_bg.pack(fill="x", padx=10, pady=(4, 4))
    fill_ratio = max(0.0, min(1.0, prob))

    def draw_bar(event):
        bar_bg.delete("all")
        width = max(event.width, 1)
        bar_bg.create_rectangle(0, 0, width, 14, fill="#1f2937", outline="")
        bar_bg.create_rectangle(
            0, 0, int(width * fill_ratio), 14, fill=probability_color(prob), outline=""
        )

    bar_bg.bind("<Configure>", draw_bar)

    tk.Label(
        card,
        text=margin_text,
        bg="#111827",
        fg="#cbd5e1",
        font=("Arial", 10),
        anchor="w",
    ).pack(fill="x", padx=10, pady=(0, 8))


def add_image_panel(parent, title: str, img: Image.Image, max_h: int = 300):
    panel = tk.Frame(parent, bg="#0f172a", highlightbackground="#334155", highlightthickness=1)
    panel.pack(side="left", padx=5, pady=5)

    tk.Label(
        panel,
        text=title,
        bg="#0f172a",
        fg="#f8fafc",
        font=("Arial", 10, "bold"),
    ).pack(fill="x", padx=6, pady=(6, 4))

    disp = resize_for_display(img, max_h=max_h)
    photo = ImageTk.PhotoImage(disp)
    label = tk.Label(panel, image=photo, bg="#0f172a")
    label.image = photo
    label.pack(padx=6, pady=(0, 6))
    return photo


def show_result_tk(
    image_path: Path,
    combined_prob: float,
    combined_margin: float,
    representative_prob: float,
    representative_margin: float,
    vit_prob: float,
    vit_margin: float,
    eva_prob: float,
    eva_margin: float,
    orig_img: Image.Image,
    vit_heat: Image.Image,
    vit_overlay: Image.Image,
    eva_heat: Image.Image,
    eva_overlay: Image.Image,
    terminal_output: str = "",
):
    root = tk.Tk()
    root.title("ViT + EVA02 Median Ensemble Result")
    root.configure(bg="#020617")
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = min(1500, max(1100, int(screen_w * 0.96)))
    win_h = min(940, max(760, int(screen_h * 0.92)))
    root.geometry(f"{win_w}x{win_h}+20+20")
    try:
        root.attributes("-zoomed", True)
    except tk.TclError:
        pass

    pred_label, pred_class = prob_margin_to_class(combined_margin)
    pred_bg = prediction_color(pred_class)

    header = tk.Frame(root, bg=pred_bg)
    header.pack(fill="x")
    tk.Label(
        header,
        text=f"Median Ensemble Prediction: {pred_class}",
        bg=pred_bg,
        fg="white",
        font=("Arial", 22, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(10, 0))
    tk.Label(
        header,
        text=(
            f"folder median result    label={pred_label}    "
            f"median p(0_sens)={combined_prob:.4f}    "
            f"median logit_margin={combined_margin:+.4f}    "
            f"representative={image_path.name}"
        ),
        bg=pred_bg,
        fg="white",
        font=("Arial", 12, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(2, 10))

    score_frame = tk.Frame(root, bg="#020617")
    score_frame.pack(fill="x", padx=8, pady=8)
    add_score_card(
        score_frame,
        "ViT-CLS",
        f"p={vit_prob:.4f}",
        vit_prob,
        f"logit_margin={vit_margin:+.4f}  weight={VIT_SPEC.weight:.2f}",
    )
    add_score_card(
        score_frame,
        "EVA02-AVG",
        f"p={eva_prob:.4f}",
        eva_prob,
        f"logit_margin={eva_margin:+.4f}  weight={EVA02_SPEC.weight:.2f}",
    )
    add_score_card(
        score_frame,
        "Folder Median",
        f"p={combined_prob:.4f}",
        combined_prob,
        f"logit_margin={combined_margin:+.4f}",
    )
    add_score_card(
        score_frame,
        "Representative Image",
        f"p={representative_prob:.4f}",
        representative_prob,
        f"logit_margin={representative_margin:+.4f}",
    )

    image_area = tk.Frame(root, bg="#020617")
    image_area.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    top_row = tk.Frame(image_area, bg="#020617")
    top_row.pack(fill="x")
    image_max_h = min(285, max(170, int((win_w - 120) / 5)), max(170, int(win_h * 0.30)))
    panels = [
        ("Original", orig_img),
        ("ViT-CLS Heatmap", vit_heat),
        ("ViT-CLS Overlay", vit_overlay),
        ("EVA02-AVG Heatmap", eva_heat),
        ("EVA02-AVG Overlay", eva_overlay),
    ]
    root._img_refs = []
    for title, img in panels:
        root._img_refs.append(add_image_panel(top_row, title, img, max_h=image_max_h))

    log_frame = tk.LabelFrame(
        root,
        text="Terminal output",
        bg="#020617",
        fg="#e5e7eb",
        font=("Arial", 10, "bold"),
        highlightbackground="#334155",
        highlightthickness=1,
    )
    log_frame.pack(fill="both", expand=True, padx=8, pady=8)
    log_text = scrolledtext.ScrolledText(
        log_frame,
        height=10,
        width=150,
        font=("Courier New", 10),
        bg="#0b1120",
        fg="#d1d5db",
        insertbackground="#e5e7eb",
        wrap="none",
    )
    log_text.pack(fill="both", expand=True, padx=6, pady=6)
    log_text.insert("1.0", terminal_output)
    log_text.configure(state="disabled")
    root.mainloop()


def main():
    log_print, terminal_logs = make_dual_printer()
    target_folder = select_target_folder()

    log_print("===== load models =====")
    vit_backend = load_backend(VIT_SPEC, DEVICE)
    eva_backend = load_backend(EVA02_SPEC, DEVICE)

    for backend in (vit_backend, eva_backend):
        spec = backend["spec"]
        input_dim = backend["models"][0].net[0].in_features
        log_print(f"[{spec.name}]")
        log_print(f"  feature : {spec.feature_model_name}")
        log_print(f"  img_size: {spec.img_size}x{spec.img_size}")
        log_print(f"  pooling : {spec.pooling}")
        log_print(f"  n_models: {len(backend['models'])}")
        log_print(f"  in_dim  : {input_dim}")
        log_print(f"  cutoff  : {backend['cutoff']:.4f}")
        log_print(f"  weight  : {spec.weight:.3f}")

    log_print(f"device   : {DEVICE}")
    log_print(f"folder   : {target_folder}")

    log_print("\n===== collect image files =====")
    image_files = collect_image_files(target_folder)
    log_print(f"n_images : {len(image_files)}")

    rows = []
    log_print("\n===== per-image dual ensemble inference =====")
    for i, image_path in enumerate(image_files, start=1):
        try:
            img = Image.open(image_path).convert("RGB")
            cropped_img = center_crop_max_square(img)
            vit_prob, vit_margin, _ = predict_with_backend(vit_backend, cropped_img)
            eva_prob, eva_margin, _ = predict_with_backend(eva_backend, cropped_img)
            combined_prob, combined_margin = combine_predictions(
                vit_prob, vit_margin, eva_prob, eva_margin
            )
        except Exception as e:
            log_print(f"[{i:03d}/{len(image_files):03d}] {image_path.name} | SKIP | {e}")
            continue

        pred_label, pred_class = prob_margin_to_class(combined_margin)
        rows.append(
            {
                "path": image_path,
                "vit_prob": vit_prob,
                "vit_margin": vit_margin,
                "eva_prob": eva_prob,
                "eva_margin": eva_margin,
                "combined_prob": combined_prob,
                "combined_margin": combined_margin,
                "pred_label": pred_label,
                "pred_class": pred_class,
            }
        )

        agreement = "agree" if (vit_margin >= 0) == (eva_margin >= 0) else "disagree"
        log_print(
            f"[{i:03d}/{len(image_files):03d}] "
            f"{image_path.name} | "
            f"pred={pred_class} | "
            f"ens_p={combined_prob:.4f} | "
            f"ens_logit_margin={combined_margin:+.4f} | "
            f"vit_p={vit_prob:.4f} ({vit_margin:+.4f}) | "
            f"eva_p={eva_prob:.4f} ({eva_margin:+.4f}) | "
            f"{agreement}"
        )

    if not rows:
        raise RuntimeError("有効な画像を1件も処理できませんでした。")

    combined_probs = np.array([r["combined_prob"] for r in rows], dtype=np.float32)
    combined_margins = np.array([r["combined_margin"] for r in rows], dtype=np.float32)
    mean_prob = float(np.mean(combined_probs))
    median_prob = float(np.median(combined_probs))
    mean_margin = float(np.mean(combined_margins))
    median_margin = float(np.median(combined_margins))
    mean_label, mean_class = prob_margin_to_class(mean_margin)
    median_label, median_class = prob_margin_to_class(median_margin)

    disagree_count = sum((r["vit_margin"] >= 0) != (r["eva_margin"] >= 0) for r in rows)

    log_print("\n===== folder summary =====")
    log_print(f"n_valid_images = {len(rows)}")
    log_print(f"n_disagree     = {disagree_count}")
    log_print("\n[MEAN ensemble]")
    log_print(f"pred_label  : {mean_label}")
    log_print(f"pred_class  : {mean_class}")
    log_print(f"prob_0_sens : {mean_prob:.4f}")
    log_print(f"logit_margin: {mean_margin:+.4f}")
    log_print("\n[MEDIAN ensemble]")
    log_print(f"pred_label  : {median_label}")
    log_print(f"pred_class  : {median_class}")
    log_print(f"prob_0_sens : {median_prob:.4f}")
    log_print(f"logit_margin: {median_margin:+.4f}")
    log_print("\n===== FINAL median ensemble decision =====")
    log_print(f"final_pred_label  : {median_label}")
    log_print(f"final_pred_class  : {median_class}")
    log_print(f"final_prob_0_sens : {median_prob:.4f}")
    log_print(f"final_logit_margin: {median_margin:+.4f}")

    med_idx = int(np.argmin(np.abs(combined_margins - median_margin)))
    selected = rows[med_idx]
    selected_path = selected["path"]

    log_print("\n===== median-margin image =====")
    log_print(f"selected_image : {selected_path}")
    log_print(f"ensemble p(0_sens): {selected['combined_prob']:.4f}")
    log_print(f"ensemble logit_margin: {selected['combined_margin']:+.4f}")

    orig_img = Image.open(selected_path).convert("RGB")
    cropped_img = center_crop_max_square(orig_img)
    vit_rel = extract_token_relevance_map(vit_backend, cropped_img)
    eva_rel = extract_token_relevance_map(eva_backend, cropped_img)
    vit_heat, vit_overlay = make_heatmap_and_overlay(cropped_img, vit_rel, alpha=0.45)
    eva_heat, eva_overlay = make_heatmap_and_overlay(cropped_img, eva_rel, alpha=0.45)

    show_result_tk(
        image_path=selected_path,
        combined_prob=median_prob,
        combined_margin=median_margin,
        representative_prob=selected["combined_prob"],
        representative_margin=selected["combined_margin"],
        vit_prob=selected["vit_prob"],
        vit_margin=selected["vit_margin"],
        eva_prob=selected["eva_prob"],
        eva_margin=selected["eva_margin"],
        orig_img=cropped_img,
        vit_heat=vit_heat,
        vit_overlay=vit_overlay,
        eva_heat=eva_heat,
        eva_overlay=eva_overlay,
        terminal_output="".join(terminal_logs),
    )


if __name__ == "__main__":
    main()
