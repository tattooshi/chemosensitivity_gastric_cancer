#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
保存済みViT-CLS ensembleモデルとcutoffを使い、フォルダー内画像のmedianで推論する単独版。

- ViT-Large / 616px / CLS特徴 / top5 ensemble
- 入力画像は中央最大正方形にcropしてから推論
- 判定は保存済み cutoff を基準にした logit margin で行う
- 006 本体は変更しない
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from PIL import Image


BASE_PATH = Path(__file__).with_name(
    "002_画像から感受性推論AIヒートマップ_tile_level_ensemble.py"
)
ENSEMBLE_INFO_PATH = "ViT616_model/top5_ensemble_info.pth"


def load_base_module():
    spec = importlib.util.spec_from_file_location("dual_ensemble_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"006の読み込みに失敗しました: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()
BACKEND_SPEC = replace(base.VIT_SPEC, ensemble_info_path=ENSEMBLE_INFO_PATH)


def margin_to_class(margin: float):
    pred_label = int(margin >= 0.0)
    pred_class = "0_sens" if pred_label == 1 else "2_resis"
    return pred_label, pred_class


def prediction_color(pred_class: str) -> str:
    return "#16a34a" if pred_class == "0_sens" else "#dc2626"


def show_result_tk(
    image_path: Path,
    prob_0_sens: float,
    margin: float,
    cutoff: float,
    representative_prob: float,
    representative_margin: float,
    orig_img: Image.Image,
    heat_img: Image.Image,
    overlay_img: Image.Image,
    terminal_output: str = "",
):
    root = tk.Tk()
    root.title("ViT-CLS Median Result")
    root.configure(bg="#020617")
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = min(1300, max(980, int(screen_w * 0.92)))
    win_h = min(900, max(720, int(screen_h * 0.90)))
    root.geometry(f"{win_w}x{win_h}+20+20")
    try:
        root.attributes("-zoomed", True)
    except tk.TclError:
        pass

    pred_label, pred_class = margin_to_class(margin)
    pred_bg = prediction_color(pred_class)

    header = tk.Frame(root, bg=pred_bg)
    header.pack(fill="x")
    tk.Label(
        header,
        text=f"ViT-CLS Median Prediction: {pred_class}",
        bg=pred_bg,
        fg="white",
        font=("Arial", 22, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(10, 0))
    tk.Label(
        header,
        text=(
            f"folder median result    label={pred_label}    "
            f"median p(0_sens)={prob_0_sens:.4f}    cutoff={cutoff:.4f}    "
            f"median logit_margin={margin:+.4f}    representative={image_path.name}"
        ),
        bg=pred_bg,
        fg="white",
        font=("Arial", 12, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(2, 10))

    score_frame = tk.Frame(root, bg="#020617")
    score_frame.pack(fill="x", padx=8, pady=8)
    base.add_score_card(
        score_frame,
        f"Folder Median  cutoff: {cutoff:.4f}",
        f"p={prob_0_sens:.4f}",
        prob_0_sens,
        f"cutoff={cutoff:.4f}  logit_margin={margin:+.4f}",
    )
    base.add_score_card(
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
    image_max_h = min(360, max(220, int((win_w - 90) / 3)), max(220, int(win_h * 0.38)))
    root._img_refs = [
        base.add_image_panel(top_row, "Original", orig_img, max_h=image_max_h),
        base.add_image_panel(top_row, "ViT-CLS Heatmap", heat_img, max_h=image_max_h),
        base.add_image_panel(top_row, "ViT-CLS Overlay", overlay_img, max_h=image_max_h),
    ]

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
    log_print, terminal_logs = base.make_dual_printer()
    target_folder = base.select_target_folder()

    log_print("===== load ViT-CLS models =====")
    backend = base.load_backend(BACKEND_SPEC, base.DEVICE)
    input_dim = backend["models"][0].net[0].in_features
    log_print(f"[{BACKEND_SPEC.name}]")
    log_print(f"  feature : {BACKEND_SPEC.feature_model_name}")
    log_print(f"  img_size: {BACKEND_SPEC.img_size}x{BACKEND_SPEC.img_size}")
    log_print(f"  pooling : {BACKEND_SPEC.pooling}")
    log_print(f"  n_models: {len(backend['models'])}")
    log_print(f"  in_dim  : {input_dim}")
    log_print(f"  cutoff  : {backend['cutoff']:.4f}")
    log_print(f"  ensemble: {BACKEND_SPEC.ensemble_info_path}")
    log_print(f"device   : {base.DEVICE}")
    log_print(f"folder   : {target_folder}")

    log_print("\n===== collect image files =====")
    image_files = base.collect_image_files(target_folder)
    log_print(f"n_images : {len(image_files)}")

    rows = []
    log_print("\n===== per-image ViT-CLS inference =====")
    for i, image_path in enumerate(image_files, start=1):
        try:
            img = Image.open(image_path).convert("RGB")
            cropped_img = base.center_crop_max_square(img)
            prob_0_sens, margin, _ = base.predict_with_backend(backend, cropped_img)
        except Exception as e:
            log_print(f"[{i:03d}/{len(image_files):03d}] {image_path.name} | SKIP | {e}")
            continue

        pred_label, pred_class = margin_to_class(margin)
        rows.append(
            {
                "path": image_path,
                "prob_0_sens": prob_0_sens,
                "margin": margin,
                "pred_label": pred_label,
                "pred_class": pred_class,
            }
        )
        log_print(
            f"[{i:03d}/{len(image_files):03d}] "
            f"{image_path.name} | "
            f"pred={pred_class} | "
            f"p(0_sens)={prob_0_sens:.4f} | "
            f"cutoff={backend['cutoff']:.4f} | "
            f"logit_margin={margin:+.4f}"
        )

    if not rows:
        raise RuntimeError("有効な画像を1件も処理できませんでした。")

    probs = np.array([r["prob_0_sens"] for r in rows], dtype=np.float32)
    margins = np.array([r["margin"] for r in rows], dtype=np.float32)
    mean_prob = float(np.mean(probs))
    median_prob = float(np.median(probs))
    mean_margin = float(np.mean(margins))
    median_margin = float(np.median(margins))
    mean_label, mean_class = margin_to_class(mean_margin)
    median_label, median_class = margin_to_class(median_margin)

    log_print("\n===== folder summary =====")
    log_print(f"cutoff = {backend['cutoff']:.4f}")
    log_print(f"n_valid_images = {len(rows)}")
    log_print("\n[MEAN ViT-CLS]")
    log_print(f"pred_label  : {mean_label}")
    log_print(f"pred_class  : {mean_class}")
    log_print(f"prob_0_sens : {mean_prob:.4f}")
    log_print(f"logit_margin: {mean_margin:+.4f}")
    log_print("\n[MEDIAN ViT-CLS]")
    log_print(f"pred_label  : {median_label}")
    log_print(f"pred_class  : {median_class}")
    log_print(f"prob_0_sens : {median_prob:.4f}")
    log_print(f"logit_margin: {median_margin:+.4f}")
    log_print("\n===== FINAL median ViT-CLS decision =====")
    log_print(f"final_pred_label  : {median_label}")
    log_print(f"final_pred_class  : {median_class}")
    log_print(f"final_prob_0_sens : {median_prob:.4f}")
    log_print(f"final_logit_margin: {median_margin:+.4f}")

    med_idx = int(np.argmin(np.abs(margins - median_margin)))
    selected = rows[med_idx]
    selected_path = selected["path"]
    log_print("\n===== median-margin image =====")
    log_print(f"selected_image : {selected_path}")
    log_print(f"p(0_sens): {selected['prob_0_sens']:.4f}")
    log_print(f"logit_margin: {selected['margin']:+.4f}")

    orig_img = Image.open(selected_path).convert("RGB")
    cropped_img = base.center_crop_max_square(orig_img)
    rel_map = base.extract_token_relevance_map(backend, cropped_img)
    heat_img, overlay_img = base.make_heatmap_and_overlay(cropped_img, rel_map, alpha=0.45)

    show_result_tk(
        image_path=selected_path,
        prob_0_sens=median_prob,
        margin=median_margin,
        cutoff=backend["cutoff"],
        representative_prob=selected["prob_0_sens"],
        representative_margin=selected["margin"],
        orig_img=cropped_img,
        heat_img=heat_img,
        overlay_img=overlay_img,
        terminal_output="".join(terminal_logs),
    )


if __name__ == "__main__":
    main()
