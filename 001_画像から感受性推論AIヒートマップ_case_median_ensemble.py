#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ViT616 と EVA02 をそれぞれフォルダー内画像の median で case 推論し、
その case-level median 結果を 009 と同じ combine_predictions() で最終判定する。

- ViT616: 各画像の p(0_sens), logit margin から median を計算して case class を推論
- EVA02 : 各画像の p(0_sens), logit margin から median を計算して case class を推論
- 最終判定: ViT median と EVA02 median を 009 と同じ重み・veto で合成
- 表示画像: ViT の median logit margin に最も近い画像
"""

import importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext

import numpy as np
from PIL import Image


BASE_PATH = Path(__file__).with_name("002_画像から感受性推論AIヒートマップ_tile_level_ensemble.py")


def load_base_module():
    spec = importlib.util.spec_from_file_location("dual_ensemble_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"009の読み込みに失敗しました: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base_module()


def show_result_tk(
    image_path: Path,
    final_prob: float,
    final_margin: float,
    vit_median_prob: float,
    vit_median_margin: float,
    vit_cutoff: float,
    eva_median_prob: float,
    eva_median_margin: float,
    eva_cutoff: float,
    representative_vit_prob: float,
    representative_vit_margin: float,
    representative_eva_prob: float,
    representative_eva_margin: float,
    orig_img: Image.Image,
    vit_heat: Image.Image,
    vit_overlay: Image.Image,
    eva_heat: Image.Image,
    eva_overlay: Image.Image,
    terminal_output: str = "",
):
    root = tk.Tk()
    root.title("Case Median ViT + EVA02 Ensemble Result")
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

    final_label, final_class = base.prob_margin_to_class(final_margin)
    vit_label, vit_class = base.prob_margin_to_class(vit_median_margin)
    eva_label, eva_class = base.prob_margin_to_class(eva_median_margin)
    final_cutoff = float(np.median([vit_cutoff, eva_cutoff]))
    pred_bg = base.prediction_color(final_class)

    header = tk.Frame(root, bg=pred_bg)
    header.pack(fill="x")
    tk.Label(
        header,
        text=f"Case Median Ensemble Prediction: {final_class}",
        bg=pred_bg,
        fg="white",
        font=("Arial", 22, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(10, 0))
    tk.Label(
        header,
        text=(
            f"final_label={final_label}    final p(0_sens)={final_prob:.4f}    "
            f"final logit_margin={final_margin:+.4f}    "
            f"ViT={vit_class}({vit_label})    EVA02={eva_class}({eva_label})    "
            f"representative={image_path.name}"
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
        f"ViT-CLS Case Median  cutoff: {vit_cutoff:.4f}",
        f"p={vit_median_prob:.4f}",
        vit_median_prob,
        f"class={vit_class}  logit_margin={vit_median_margin:+.4f}",
    )
    base.add_score_card(
        score_frame,
        f"EVA02-AVG Case Median  cutoff: {eva_cutoff:.4f}",
        f"p={eva_median_prob:.4f}",
        eva_median_prob,
        f"class={eva_class}  logit_margin={eva_median_margin:+.4f}",
    )
    base.add_score_card(
        score_frame,
        f"Final Ensemble  cutoff: {final_cutoff:.4f}",
        f"p={final_prob:.4f}",
        final_prob,
        f"logit_margin={final_margin:+.4f}",
    )
    base.add_score_card(
        score_frame,
        "ViT Median Image",
        f"p={representative_vit_prob:.4f}",
        representative_vit_prob,
        (
            f"vit_margin={representative_vit_margin:+.4f}  "
            f"eva_p={representative_eva_prob:.4f}  eva_margin={representative_eva_margin:+.4f}"
        ),
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
        root._img_refs.append(base.add_image_panel(top_row, title, img, max_h=image_max_h))

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

    log_print("===== load models =====")
    vit_backend = base.load_backend(base.VIT_SPEC, base.DEVICE)
    eva_backend = base.load_backend(base.EVA02_SPEC, base.DEVICE)

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

    log_print(f"device   : {base.DEVICE}")
    log_print(f"folder   : {target_folder}")

    log_print("\n===== collect image files =====")
    image_files = base.collect_image_files(target_folder)
    log_print(f"n_images : {len(image_files)}")

    rows = []
    log_print("\n===== per-image ViT / EVA02 inference =====")
    for i, image_path in enumerate(image_files, start=1):
        try:
            img = Image.open(image_path).convert("RGB")
            cropped_img = base.center_crop_max_square(img)
            vit_prob, vit_margin, _ = base.predict_with_backend(vit_backend, cropped_img)
            eva_prob, eva_margin, _ = base.predict_with_backend(eva_backend, cropped_img)
        except Exception as e:
            log_print(f"[{i:03d}/{len(image_files):03d}] {image_path.name} | SKIP | {e}")
            continue

        vit_label, vit_class = base.prob_margin_to_class(vit_margin)
        eva_label, eva_class = base.prob_margin_to_class(eva_margin)
        rows.append(
            {
                "path": image_path,
                "vit_prob": vit_prob,
                "vit_margin": vit_margin,
                "vit_label": vit_label,
                "vit_class": vit_class,
                "eva_prob": eva_prob,
                "eva_margin": eva_margin,
                "eva_label": eva_label,
                "eva_class": eva_class,
            }
        )
        agreement = "agree" if vit_label == eva_label else "disagree"
        log_print(
            f"[{i:03d}/{len(image_files):03d}] "
            f"{image_path.name} | "
            f"vit={vit_class} p={vit_prob:.4f} margin={vit_margin:+.4f} | "
            f"eva={eva_class} p={eva_prob:.4f} margin={eva_margin:+.4f} | "
            f"{agreement}"
        )

    if not rows:
        raise RuntimeError("有効な画像を1件も処理できませんでした。")

    vit_probs = np.array([r["vit_prob"] for r in rows], dtype=np.float32)
    vit_margins = np.array([r["vit_margin"] for r in rows], dtype=np.float32)
    eva_probs = np.array([r["eva_prob"] for r in rows], dtype=np.float32)
    eva_margins = np.array([r["eva_margin"] for r in rows], dtype=np.float32)

    vit_mean_prob = float(np.mean(vit_probs))
    vit_mean_margin = float(np.mean(vit_margins))
    vit_median_prob = float(np.median(vit_probs))
    vit_median_margin = float(np.median(vit_margins))
    eva_mean_prob = float(np.mean(eva_probs))
    eva_mean_margin = float(np.mean(eva_margins))
    eva_median_prob = float(np.median(eva_probs))
    eva_median_margin = float(np.median(eva_margins))

    vit_median_label, vit_median_class = base.prob_margin_to_class(vit_median_margin)
    eva_median_label, eva_median_class = base.prob_margin_to_class(eva_median_margin)
    final_prob, final_margin = base.combine_predictions(
        vit_median_prob,
        vit_median_margin,
        eva_median_prob,
        eva_median_margin,
    )
    final_label, final_class = base.prob_margin_to_class(final_margin)

    disagree_count = sum(r["vit_label"] != r["eva_label"] for r in rows)

    log_print("\n===== folder summary =====")
    log_print(f"n_valid_images = {len(rows)}")
    log_print(f"n_disagree     = {disagree_count}")
    log_print("\n[MEAN ViT-CLS]")
    log_print(f"prob_0_sens : {vit_mean_prob:.4f}")
    log_print(f"logit_margin: {vit_mean_margin:+.4f}")
    log_print("\n[MEDIAN ViT-CLS CASE]")
    log_print(f"pred_label  : {vit_median_label}")
    log_print(f"pred_class  : {vit_median_class}")
    log_print(f"prob_0_sens : {vit_median_prob:.4f}")
    log_print(f"logit_margin: {vit_median_margin:+.4f}")
    log_print("\n[MEAN EVA02-AVG]")
    log_print(f"prob_0_sens : {eva_mean_prob:.4f}")
    log_print(f"logit_margin: {eva_mean_margin:+.4f}")
    log_print("\n[MEDIAN EVA02-AVG CASE]")
    log_print(f"pred_label  : {eva_median_label}")
    log_print(f"pred_class  : {eva_median_class}")
    log_print(f"prob_0_sens : {eva_median_prob:.4f}")
    log_print(f"logit_margin: {eva_median_margin:+.4f}")
    log_print("\n===== FINAL case-median ensemble decision =====")
    log_print(f"final_pred_label  : {final_label}")
    log_print(f"final_pred_class  : {final_class}")
    log_print(f"final_prob_0_sens : {final_prob:.4f}")
    log_print(f"final_logit_margin: {final_margin:+.4f}")

    med_idx = int(np.argmin(np.abs(vit_margins - vit_median_margin)))
    selected = rows[med_idx]
    selected_path = selected["path"]
    log_print("\n===== ViT median-margin image =====")
    log_print(f"selected_image : {selected_path}")
    log_print(f"vit p(0_sens): {selected['vit_prob']:.4f}")
    log_print(f"vit logit_margin: {selected['vit_margin']:+.4f}")
    log_print(f"eva p(0_sens): {selected['eva_prob']:.4f}")
    log_print(f"eva logit_margin: {selected['eva_margin']:+.4f}")

    orig_img = Image.open(selected_path).convert("RGB")
    cropped_img = base.center_crop_max_square(orig_img)
    vit_rel = base.extract_token_relevance_map(vit_backend, cropped_img)
    eva_rel = base.extract_token_relevance_map(eva_backend, cropped_img)
    vit_heat, vit_overlay = base.make_heatmap_and_overlay(cropped_img, vit_rel, alpha=0.45)
    eva_heat, eva_overlay = base.make_heatmap_and_overlay(cropped_img, eva_rel, alpha=0.45)

    show_result_tk(
        image_path=selected_path,
        final_prob=final_prob,
        final_margin=final_margin,
        vit_median_prob=vit_median_prob,
        vit_median_margin=vit_median_margin,
        vit_cutoff=vit_backend["cutoff"],
        eva_median_prob=eva_median_prob,
        eva_median_margin=eva_median_margin,
        eva_cutoff=eva_backend["cutoff"],
        representative_vit_prob=selected["vit_prob"],
        representative_vit_margin=selected["vit_margin"],
        representative_eva_prob=selected["eva_prob"],
        representative_eva_margin=selected["eva_margin"],
        orig_img=cropped_img,
        vit_heat=vit_heat,
        vit_overlay=vit_overlay,
        eva_heat=eva_heat,
        eva_overlay=eva_overlay,
        terminal_output="".join(terminal_logs),
    )


if __name__ == "__main__":
    main()
