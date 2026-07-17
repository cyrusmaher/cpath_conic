from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.segmentation import find_boundaries

from .constants import CLASS_COLORS, CLASS_NAMES, SHORT_COUNT_COLUMNS
from .data import central_crop_counts, load_patch


ERROR_COLORS = {
    "Missed GT (FN)": (255, 40, 40),
    "Extra prediction (FP)": (40, 120, 255),
    "Wrong class": (255, 190, 0),
    "Correct overlap": (65, 190, 80),
}


def _font(size: int = 14, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size=size)
    except OSError:
        return ImageFont.load_default()


def _labeled_tile(array: np.ndarray, label: str, bar_height: int = 25) -> np.ndarray:
    tile = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(tile, "RGBA")
    draw.rectangle((0, 0, tile.width, bar_height), fill=(0, 0, 0, 205))
    draw.text((7, 4), label, fill=(255, 255, 255, 255), font=_font(13, bold=True))
    return np.asarray(tile)


def _legend_strip(width: int, include_errors: bool = True, scale: int = 1) -> np.ndarray:
    class_items = [(name, CLASS_COLORS[index]) for index, name in enumerate(CLASS_NAMES, start=1)]
    error_items = list(ERROR_COLORS.items()) if include_errors else []
    row_height = 26 * scale
    height = row_height * (2 if error_items else 1) + 8 * scale
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    font = _font(11 * scale)

    def draw_row(items, y, prefix):
        draw.text((7 * scale, y + 5 * scale), prefix, fill=(25, 25, 25), font=_font(11 * scale, bold=True))
        x = (88 if prefix == "Cell class" else 108) * scale
        for name, color in items:
            swatch = 11 * scale
            draw.rectangle((x, y + 5 * scale, x + swatch, y + 5 * scale + swatch), fill=tuple(color), outline=(35, 35, 35))
            x += swatch + 4 * scale
            draw.text((x, y + 3 * scale), name, fill=(25, 25, 25), font=font)
            x += int(draw.textlength(name, font=font)) + 13 * scale

    draw_row(class_items, 2 * scale, "Cell class")
    if error_items:
        draw_row(error_items, 2 * scale + row_height, "Comparison")
    return np.asarray(canvas)


def _gif_frame(array: np.ndarray, title: str, include_errors: bool = False, scale: int = 3) -> np.ndarray:
    image = Image.fromarray(array.astype(np.uint8)).convert("RGB").resize(
        (array.shape[1] * scale, array.shape[0] * scale), Image.Resampling.NEAREST
    )
    header_height = 46
    legend = _legend_strip(image.width, include_errors=include_errors, scale=1)
    canvas = Image.new("RGB", (image.width, header_height + image.height + legend.shape[0]), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 11), title, fill=(255, 255, 255), font=_font(20, bold=True))
    canvas.paste(image, (0, header_height))
    canvas.paste(Image.fromarray(legend), (0, header_height + image.height))
    return np.asarray(canvas)


def _save_gif(frames: list[np.ndarray], path: Path, durations_ms: list[int]) -> None:
    """Save a shared-palette GIF while reserving exact semantic overlay colors."""
    semantic_colors = list(CLASS_COLORS.values()) + list(ERROR_COLORS.values())
    interface_colors = [(0, 0, 0), (255, 255, 255), (248, 248, 248), (20, 20, 20)]
    required = list(dict.fromkeys(tuple(color) for color in semantic_colors + interface_colors))
    rgb_frames = [Image.fromarray(frame.astype(np.uint8)).convert("RGB") for frame in frames]
    sample = Image.new("RGB", (rgb_frames[0].width, rgb_frames[0].height * len(rgb_frames)))
    for index, frame in enumerate(rgb_frames):
        sample.paste(frame, (0, index * frame.height))
    n_adaptive = 256 - len(required)
    adaptive = sample.quantize(colors=n_adaptive, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    adaptive_palette = adaptive.getpalette()[: n_adaptive * 3]
    generated = [tuple(adaptive_palette[index : index + 3]) for index in range(0, len(adaptive_palette), 3)]
    colors = required + [color for color in generated if color not in required]
    colors = colors[:256]
    colors.extend([(0, 0, 0)] * (256 - len(colors)))
    palette_image = Image.new("P", (1, 1))
    palette_image.putpalette([component for color in colors for component in color])
    indexed = [frame.quantize(palette=palette_image, dither=Image.Dither.NONE) for frame in rgb_frames]
    indexed[0].save(
        path,
        save_all=True,
        append_images=indexed[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )


def colorize(class_map: np.ndarray) -> np.ndarray:
    out = np.zeros((*class_map.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        out[class_map == cls] = color
    return out


def instance_boundaries(inst_map: np.ndarray, class_map: np.ndarray) -> np.ndarray:
    out = colorize(class_map)
    boundary = find_boundaries(inst_map, mode="outer")
    out[boundary] = np.asarray([255, 255, 255], dtype=np.uint8)
    return out


def overlay(image: np.ndarray, mask_rgb: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    return np.clip((1 - alpha) * image.astype(float) + alpha * mask_rgb.astype(float), 0, 255).astype(np.uint8)


def error_overlay(true_inst, true_cls, pred_inst, pred_cls) -> np.ndarray:
    out = np.zeros((*true_inst.shape, 3), dtype=np.uint8)
    out[(true_inst > 0) & (pred_inst == 0)] = ERROR_COLORS["Missed GT (FN)"]
    out[(true_inst == 0) & (pred_inst > 0)] = ERROR_COLORS["Extra prediction (FP)"]
    out[(true_inst > 0) & (pred_inst > 0) & (true_cls != pred_cls)] = ERROR_COLORS["Wrong class"]
    out[(true_inst > 0) & (pred_inst > 0) & (true_cls == pred_cls)] = ERROR_COLORS["Correct overlap"]
    return out


def _bar_figure(gt: np.ndarray, pred: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 2.7), dpi=120)
    x = np.arange(len(CLASS_NAMES))
    ax.bar(x - 0.18, gt, 0.36, label="GT", color="#666666")
    ax.bar(x + 0.18, pred, 0.36, label="Prediction", color="#3f7fbf")
    ax.set_xticks(x, [n[:5] for n in CLASS_NAMES], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("nuclei")
    ax.set_title(title, fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def render_panel(
    image,
    true_inst,
    true_cls,
    pred_inst,
    pred_cls,
    gt_counts,
    pred_counts,
    title: str,
    count_method: str = "mask-derived counts",
) -> np.ndarray:
    gt_rgb = instance_boundaries(true_inst, true_cls)
    pred_rgb = instance_boundaries(pred_inst, pred_cls)
    err_rgb = error_overlay(true_inst, true_cls, pred_inst, pred_cls)
    count_rgb = _bar_figure(gt_counts, pred_counts, f"Central-region counts · {count_method}")
    gt_overlay = overlay(image, gt_rgb)
    pred_overlay = overlay(image, pred_rgb)
    comparison_overlay = overlay(image, err_rgb, alpha=0.52)
    count_tile = np.asarray(Image.fromarray(count_rgb).resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS))
    top = np.concatenate([
        _labeled_tile(count_tile, f"Central-region counts · {count_method}"),
        _labeled_tile(gt_rgb, "Ground-truth instances"),
        _labeled_tile(pred_rgb, "Predicted instances"),
        _labeled_tile(err_rgb, "Pixel comparison"),
    ], axis=1)
    bottom = np.concatenate([
        _labeled_tile(image, "Original H&E"),
        _labeled_tile(gt_overlay, "Ground-truth overlay"),
        _labeled_tile(pred_overlay, "Prediction overlay"),
        _labeled_tile(comparison_overlay, "Comparison overlay"),
    ], axis=1)
    grid = np.concatenate([top, bottom], axis=0)
    legend = _legend_strip(grid.shape[1], include_errors=True)
    header_height = 31
    canvas = Image.new("RGB", (grid.shape[1], header_height + grid.shape[0] + legend.shape[0]), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(255, 255, 255), font=_font(14, bold=True))
    canvas.paste(Image.fromarray(grid), (0, header_height))
    canvas.paste(Image.fromarray(legend), (0, header_height + grid.shape[0]))
    return np.asarray(canvas)


def render_cutout_strip(
    image: np.ndarray,
    true_inst: np.ndarray,
    true_cls: np.ndarray,
    pred_inst: np.ndarray,
    pred_cls: np.ndarray,
    probabilities: dict[int, np.ndarray],
    patch_id: int,
    outdir: Path,
    max_per_class: int = 12,
) -> tuple[Path, list[dict]]:
    """Render GT-selected nucleus cutouts sorted by P(ground-truth class)."""
    tile = 128
    label_h = 34
    gap = 7
    width = max_per_class * (tile + gap) + gap
    height = len(CLASS_NAMES) * (tile + label_h + gap) + gap
    canvas = Image.new("RGB", (width, height), (238, 238, 238))
    draw = ImageDraw.Draw(canvas)
    records = []
    nonempty_class_ids = []
    for class_id, class_name in enumerate(CLASS_NAMES, start=1):
        candidates = []
        for gt_instance_id in np.unique(true_inst):
            if gt_instance_id == 0:
                continue
            gt_mask = true_inst == gt_instance_id
            gt_pixels = true_cls[gt_mask]
            gt_class_id = int(np.bincount(gt_pixels.astype(np.int64), minlength=7).argmax())
            if gt_class_id != class_id:
                continue
            overlap_ids, overlap_counts = np.unique(pred_inst[gt_mask], return_counts=True)
            overlap = [(int(count), int(pred_id)) for pred_id, count in zip(overlap_ids, overlap_counts) if pred_id != 0]
            pred_instance_id = 0
            match_iou = 0.0
            if overlap:
                intersection, pred_instance_id = max(overlap)
                pred_mask = pred_inst == pred_instance_id
                union = int(gt_mask.sum() + pred_mask.sum() - intersection)
                match_iou = intersection / union if union else 0.0
            if pred_instance_id:
                pred_pixels = pred_cls[pred_inst == pred_instance_id]
                pred_class_id = int(np.bincount(pred_pixels.astype(np.int64), minlength=7).argmax())
                fallback = np.eye(6, dtype=np.float32)[max(1, pred_class_id) - 1]
                probs = np.asarray(probabilities.get(pred_instance_id, fallback), dtype=np.float32)
            else:
                pred_class_id = 0
                probs = np.zeros(6, dtype=np.float32)
            probability_gt = float(probs[class_id - 1])
            correct = bool(match_iou > 0.5 and pred_class_id == class_id)
            if pred_instance_id == 0:
                status = "missed"
            elif match_iou <= 0.5:
                status = "low IoU"
            elif pred_class_id != class_id:
                status = "wrong class"
            else:
                status = "correct"
            ys, xs = np.where(gt_mask)
            y0, y1 = max(0, int(ys.min()) - 1), min(image.shape[0], int(ys.max()) + 2)
            x0, x1 = max(0, int(xs.min()) - 1), min(image.shape[1], int(xs.max()) + 2)
            crop_array = image[y0:y1, x0:x1]
            mask_array = (true_inst[y0:y1, x0:x1] == gt_instance_id).astype(np.uint8) * 255
            crop = Image.fromarray(crop_array).convert("RGBA")
            crop.putalpha(Image.fromarray(mask_array, mode="L"))
            record = {
                "gt_instance_id": int(gt_instance_id),
                "gt_class": class_name,
                "matched_pred_instance_id": int(pred_instance_id),
                "matched_pred_class": CLASS_NAMES[pred_class_id - 1] if pred_class_id else "missed",
                "match_iou": round(float(match_iou), 4),
                "probability_gt_class": round(probability_gt, 6),
                "correct": correct,
                "status": status,
                "probabilities": {name: round(float(value), 6) for name, value in zip(CLASS_NAMES, probs)},
            }
            candidates.append((probability_gt, crop, record))
        candidates.sort(key=lambda item: item[0], reverse=True)
        if len(candidates) > max_per_class:
            high_count = max_per_class // 2
            selected = candidates[:high_count] + candidates[-(max_per_class - high_count) :]
            candidates_to_render = sorted(selected, key=lambda item: item[0], reverse=True)
        else:
            candidates_to_render = candidates
        if not candidates:
            continue
        nonempty_class_ids.append(class_id)
        y = gap + (class_id - 1) * (tile + label_h + gap)
        color = CLASS_COLORS[class_id]
        draw.rectangle((gap, y + 7, gap + 14, y + 21), fill=color, outline=(30, 30, 30))
        draw.text((gap + 21, y + 4), f"GT {class_name} (n={len(candidates)})", fill=(0, 0, 0), font=_font(15, bold=True))
        draw.text((width - 226, y + 6), "high → low P(GT class)", fill=(65, 65, 65), font=_font(12))
        for col, (probability_gt, crop, record) in enumerate(candidates_to_render):
            x = gap + col * (tile + gap)
            inner = tile - 16
            scale = min(inner / crop.width, inner / crop.height)
            size = (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale))))
            crop = crop.resize(size, Image.Resampling.NEAREST)
            cell = Image.new("RGB", (tile, tile), (42, 42, 42))
            position = ((tile - crop.width) // 2, (tile - crop.height) // 2)
            cell.paste(crop.convert("RGB"), position, crop.getchannel("A"))
            canvas.paste(cell, (x, y + label_h))
            status_color = (34, 197, 94) if record["correct"] else (250, 204, 21)
            draw.rectangle((x, y + label_h, x + tile - 1, y + label_h + tile - 1), outline=status_color, width=5)
            draw.text((x + 7, y + label_h + 6), f"GT#{record['gt_instance_id']}", fill=(255, 255, 255), font=_font(12, bold=True), stroke_width=2, stroke_fill=(0, 0, 0))
            status_text = (
                "correct"
                if record["correct"]
                else (
                    f"IoU={record['match_iou']:.2f}"
                    if record["status"] == "low IoU"
                    else record["status"]
                )
            )
            draw.text((x + 7, y + label_h + 27), status_text, fill=status_color, font=_font(12, bold=True), stroke_width=2, stroke_fill=(0, 0, 0))
            draw.text((x + 7, y + label_h + tile - 25), f"P(GT)={probability_gt:.3f}", fill=(255, 255, 255), font=_font(13, bold=True), stroke_width=2, stroke_fill=(0, 0, 0))
            records.append(record)
    if nonempty_class_ids:
        compact_height = len(nonempty_class_ids) * (tile + label_h + gap) + gap
        compact = Image.new("RGB", (width, compact_height), (238, 238, 238))
        row_pixels = tile + label_h
        for row_index, class_id in enumerate(nonempty_class_ids):
            source_y = gap + (class_id - 1) * (tile + label_h + gap)
            destination_y = gap + row_index * (tile + label_h + gap)
            compact.paste(canvas.crop((0, source_y, width, source_y + row_pixels)), (0, destination_y))
        canvas = compact
    path = outdir / "cutouts" / f"{patch_id:05d}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path, records


def render_case(
    prepared: Path,
    predictions: np.ndarray,
    patch_id: int,
    outdir: Path,
    split: str,
    metadata_row: dict | None = None,
    probabilities: dict[int, np.ndarray] | None = None,
    pred_counts_override: np.ndarray | None = None,
    count_method: str = "mask-derived counts",
) -> dict:
    image, true_inst, true_cls, row = load_patch(prepared, patch_id)
    prediction = predictions[patch_id]
    pred_inst, pred_cls = prediction[..., 0].astype(np.int32), prediction[..., 1].astype(np.uint8)
    gt_counts = central_crop_counts(true_inst, true_cls)
    mask_counts = central_crop_counts(pred_inst, pred_cls)
    pred_counts = (
        np.asarray(pred_counts_override, dtype=np.int64)
        if pred_counts_override is not None
        else mask_counts
    )
    if pred_counts.shape != (len(CLASS_NAMES),):
        raise ValueError(f"Expected one count per class for patch {patch_id}, got {pred_counts.shape}")
    title = f"patch {patch_id} | split={split} | source={row.get('source', '')}"
    panel = render_panel(
        image,
        true_inst,
        true_cls,
        pred_inst,
        pred_cls,
        gt_counts,
        pred_counts,
        title,
        count_method=count_method,
    )
    panels = outdir / "panels"
    animations = outdir / "animations"
    panels.mkdir(parents=True, exist_ok=True)
    animations.mkdir(parents=True, exist_ok=True)
    panel_path = panels / f"{patch_id:05d}.png"
    Image.fromarray(panel).save(panel_path)
    probability_map = probabilities or {}
    cutout_path, cutout_examples = render_cutout_strip(
        image, true_inst, true_cls, pred_inst, pred_cls, probability_map, patch_id, outdir
    )
    cell_probabilities = []
    for instance_id in np.unique(pred_inst):
        if instance_id == 0:
            continue
        ys, xs = np.where(pred_inst == instance_id)
        probs = np.asarray(probability_map.get(int(instance_id), np.eye(6, dtype=np.float32)[int(np.bincount(pred_cls[pred_inst == instance_id].astype(np.int64), minlength=7)[1:].argmax())]), dtype=float)
        cell_probabilities.append({"instance_id": int(instance_id), "centroid": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)], "predicted_class": CLASS_NAMES[int(probs.argmax())], "probabilities": {name: round(float(value), 6) for name, value in zip(CLASS_NAMES, probs)}})
    gt_overlay = overlay(image, instance_boundaries(true_inst, true_cls))
    pred_overlay = overlay(image, instance_boundaries(pred_inst, pred_cls))
    comparison = overlay(image, error_overlay(true_inst, true_cls, pred_inst, pred_cls), alpha=0.52)
    frames = [
        _gif_frame(image, "1 / 4 — Original H&E", include_errors=True),
        _gif_frame(gt_overlay, "2 / 4 — Ground-truth overlay", include_errors=True),
        _gif_frame(pred_overlay, "3 / 4 — Prediction overlay", include_errors=True),
        _gif_frame(comparison, "4 / 4 — Comparison overlay", include_errors=True),
    ]
    gif_path = animations / f"{patch_id:05d}.gif"
    _save_gif(frames, gif_path, durations_ms=[1200, 1200, 1200, 2000])
    residual = pred_counts - gt_counts
    return {
        "patch_id": int(patch_id),
        "split": split,
        "source": row.get("source", ""),
        "patch_info": row.get("patch_info", ""),
        "panel": str(panel_path.relative_to(outdir)),
        "cutouts": str(cutout_path.relative_to(outdir)),
        "animation": str(gif_path.relative_to(outdir)),
        "gt_counts": gt_counts.tolist(),
        "pred_counts": pred_counts.tolist(),
        "mask_counts": mask_counts.tolist(),
        "count_method": count_method,
        "count_error": residual.tolist(),
        "count_signed_error": float(residual.sum()),
        "count_abs_error": float(np.abs(residual).sum()),
        "cutout_examples": cutout_examples,
        "cell_probabilities": cell_probabilities,
        "notes": "",
        "human_verdict": "unreviewed",
        "error_type": "",
    }


def _metric_text(value) -> str:
    return "—" if value is None or not np.isfinite(float(value)) else f"{float(value):.4f}"


def _verdict_chip(row: dict) -> str:
    """A closed-vocabulary outcome chip. Falls back to legacy status if absent."""
    outcome = row.get("outcome")
    if not outcome:
        return f"<span class='status'>{html.escape(row.get('status', 'planned'))}</span>"
    label = html.escape(row.get("outcome_label", outcome.title()))
    hint = html.escape(row.get("outcome_hint", ""), quote=True)
    return f"<span class='chip chip-{html.escape(outcome, quote=True)}' title='{hint}'>{label}</span>"


def _provenance_chip(evaluation_set: str) -> str:
    """Tag a number by the split it was computed on, so eyes never compare across folds."""
    text = evaluation_set or "Evaluation set not recorded"
    lower = text.lower()
    if "internal test" in lower or "retrospective" in lower:
        kind, short = "test", "internal test"
    elif "external" in lower or "public fold" in lower or "authors" in lower:
        kind, short = "external", "external fold"
    elif "validation" in lower or "oof" in lower or "development" in lower:
        kind, short = "val", "dev validation"
    elif "not yet" in lower or "not evaluated" in lower:
        kind, short = "none", "not scored"
    else:
        kind, short = "other", text
    return f"<span class='prov prov-{kind}' title='{html.escape(text, quote=True)}'>{html.escape(short)}</span>"


def _mpq_diagnosis(performance: dict) -> str:
    """One-paragraph read on what drove the final mPQ+ result."""
    best_mpq = performance.get("best_mpq", {})
    dq, sq = best_mpq.get("dq"), best_mpq.get("sq")
    if dq is None or sq is None:
        return ""
    return (
        f"The selected model's matched-cell shape quality is strong (mSQ+ {sq:.3f}); the harder part is finding "
        f"and correctly typing every class (mDQ+ {dq:.3f}). That detection-and-typing gap is why the mPQ+ "
        "improvement is smaller than the count-R² improvement."
    )


def _improvement_tiles(performance: dict) -> str:
    """Baseline → selected-model improvement, as percent gains with the raw values.

    Count R² stands alone; mPQ+ carries its mDQ+/mSQ+ decomposition so the reader
    sees which component moved. Only the baseline and the selected model appear.
    """
    baseline = performance.get("baseline", {})
    final = performance.get("best_mpq", {})
    if baseline.get("r2") is None or final.get("r2") is None:
        return ""

    def pct(key: str) -> str:
        base, value = baseline.get(key), final.get(key)
        if base is None or value is None or base == 0:
            return "—", "up"
        change = (value - base) / abs(base) * 100.0
        return f"{change:+.0f}%", ("up" if change >= 0 else "down")

    def arrow(key: str) -> str:
        base, value = baseline.get(key), final.get(key)
        return f"{base:.2f} → {value:.2f}" if base is not None and value is not None else "—"

    r2_change, r2_dir = pct("r2")
    mpq_change, mpq_dir = pct("mpq")

    def sub(label: str, key: str) -> str:
        change, direction = pct(key)
        return (
            f"<div class='improve-sub-metric'><span>{html.escape(label)}</span>"
            f"<b class='{direction}'>{change}</b><span class='improve-sub-arrow'>{arrow(key)}</span></div>"
        )

    return f"""
      <div class='improvement-grid'>
        <article class='improve-tile'>
          <span class='eyebrow'>Count R²</span>
          <b class='improve-pct {r2_dir}'>{r2_change}</b>
          <span class='improve-sub'>improvement over baseline</span>
          <span class='improve-arrow'>{arrow('r2')}</span>
        </article>
        <article class='improve-tile improve-tile-wide'>
          <span class='eyebrow'>mPQ+</span>
          <b class='improve-pct {mpq_dir}'>{mpq_change}</b>
          <span class='improve-sub'>improvement over baseline</span>
          <span class='improve-arrow'>{arrow('mpq')}</span>
          <div class='improve-sub-grid'>{sub('mDQ+ detection', 'dq')}{sub('mSQ+ shape', 'sq')}</div>
        </article>
      </div>
    """


def _trajectory_panel(metric: str, title: str, subtitle: str, trajectory: dict, show_target: bool = True, recommended_id: str | None = None) -> str:
    """One small-multiple panel: every scored attempt plus the running-best frontier.

    Two metrics on different scales get two panels sharing an x axis rather than
    one dual-axis plot, whose scale alignment would be arbitrary.
    """
    points = trajectory.get("points", [])
    series = trajectory.get("series", {}).get(metric, {})
    frontier = series.get("frontier", [])
    if not points or not frontier:
        return ""
    target = series.get("target") if show_target else None

    width, height = 900, 320
    margin = {"l": 62, "r": 132, "t": 18, "b": 46}
    max_sequence = max(point["sequence"] for point in points)
    values = [point[metric] for point in points] + ([float(target)] if target else [])
    low, high = min(values), max(values)
    pad = max(0.02, (high - low) * 0.12)
    low, high = low - pad, high + pad

    def x_of(sequence: float) -> float:
        return margin["l"] + sequence / max(1, max_sequence) * (width - margin["l"] - margin["r"])

    def y_of(value: float) -> float:
        return height - margin["b"] - (value - low) / (high - low) * (height - margin["t"] - margin["b"])

    parts = [
        f"<rect x='{margin['l']}' y='{margin['t']}' width='{width - margin['l'] - margin['r']}' "
        f"height='{height - margin['t'] - margin['b']}' class='traj-plot'/>"
    ]
    for step in range(5):
        value = low + (high - low) * step / 4
        y = y_of(value)
        parts.append(f"<line x1='{margin['l']}' y1='{y:.1f}' x2='{width - margin['r']}' y2='{y:.1f}' class='traj-grid'/>")
        parts.append(f"<text x='{margin['l'] - 9}' y='{y + 4:.1f}' text-anchor='end' class='traj-tick'>{value:.2f}</text>")
    for sequence in range(0, max_sequence + 1, 5):
        x = x_of(sequence)
        parts.append(f"<text x='{x:.1f}' y='{height - margin['b'] + 19}' text-anchor='middle' class='traj-tick'>E{sequence:02d}</text>")

    if target is not None:
        y = y_of(float(target))
        parts.append(f"<line x1='{margin['l']}' y1='{y:.1f}' x2='{width - margin['r']}' y2='{y:.1f}' class='traj-target'/>")
        parts.append(
            f"<text x='{margin['l'] + 6}' y='{y - 6:.1f}' text-anchor='start' class='traj-target-label'>target {float(target):.3f}</text>"
        )

    # Every scored attempt, including the ones that never advanced the frontier.
    for point in points:
        marker = (
            f"<circle cx='{x_of(point['sequence']):.1f}' cy='{y_of(point[metric]):.1f}' r='4' class='traj-dot'>"
            f"<title>{html.escape(point['id'])} · {html.escape(point['method'])}\n"
            f"{title}: {point[metric]:.4f}\nverdict: {html.escape(point['outcome'])}</title></circle>"
        )
        parts.append(marker)

    # Running-best frontier as a step line: a later attempt only counts if it beat
    # everything before it.
    path = [f"M {x_of(frontier[0]['sequence']):.1f} {y_of(frontier[0]['value']):.1f}"]
    for previous, current in zip(frontier, frontier[1:]):
        path.append(f"L {x_of(current['sequence']):.1f} {y_of(previous['value']):.1f}")
        path.append(f"L {x_of(current['sequence']):.1f} {y_of(current['value']):.1f}")
    path.append(f"L {x_of(max_sequence):.1f} {y_of(frontier[-1]['value']):.1f}")
    parts.append(f"<path d='{' '.join(path)}' class='traj-frontier'/>")

    jumps = [
        (current["value"] - previous["value"], current)
        for previous, current in zip(frontier, frontier[1:])
    ]
    biggest = max(jumps, key=lambda item: item[0])[1] if jumps else None
    for step in frontier:
        parts.append(
            f"<circle cx='{x_of(step['sequence']):.1f}' cy='{y_of(step['value']):.1f}' r='5.5' class='traj-lead'>"
            f"<title>{html.escape(step['id'])} took the lead · {title} {step['value']:.4f}</title></circle>"
        )
    annotated = []
    for step, dy in ((biggest, -12), (frontier[-1], 18)):
        if step is None or any(other["sequence"] == step["sequence"] for other in annotated):
            continue
        annotated.append(step)
        parts.append(
            f"<text x='{x_of(step['sequence']):.1f}' y='{y_of(step['value']) + dy:.1f}' "
            f"text-anchor='middle' class='traj-annotation'>{html.escape(step['id'])} {step['value']:.3f}</text>"
        )
    # Mark the recommended model explicitly. It is the frontier endpoint for the
    # metric it leads (mPQ+), but sits below the frontier for the other (a count
    # blend reached higher R²); showing it on both panels keeps that honest.
    recommended = next((point for point in points if point["id"] == recommended_id), None) if recommended_id else None
    if recommended is not None:
        rx, ry = x_of(recommended["sequence"]), y_of(recommended[metric])
        parts.append(
            f"<circle cx='{rx:.1f}' cy='{ry:.1f}' r='7' class='traj-recommended'>"
            f"<title>Recommended model {html.escape(recommended['id'])} · {title} {recommended[metric]:.4f}</title></circle>"
        )
        if not any(other["sequence"] == recommended["sequence"] for other in annotated):
            anchor = "end" if rx > width - margin["r"] - 60 else "middle"
            parts.append(
                f"<text x='{rx:.1f}' y='{ry + 20:.1f}' text-anchor='{anchor}' class='traj-rec-label'>"
                f"{html.escape(recommended['id'])} {recommended[metric]:.3f} · recommended</text>"
            )
    return (
        f"<figure class='traj-panel'><figcaption><strong>{html.escape(title)}</strong>"
        f"<span>{html.escape(subtitle)}</span></figcaption>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='{html.escape(title)} by experiment number'>"
        f"{''.join(parts)}</svg></figure>"
    )


def _trajectory_html(performance: dict) -> str:
    """The narrative view from baseline experiments to the recommended model."""
    trajectory = performance.get("trajectory", {})
    if not trajectory.get("points"):
        return ""
    points = trajectory["points"]
    scored = len(points)
    last_sequence = max(point["sequence"] for point in points)
    r2_frontier = trajectory["series"]["r2"]["frontier"]
    mpq_frontier = trajectory["series"]["mpq"]["frontier"]
    lead_changes = len(r2_frontier) + len(mpq_frontier) - 2
    recommended_id = performance.get("best_mpq", {}).get("id")
    recommended_r2 = performance.get("best_mpq", {}).get("r2")
    r2_peak = r2_frontier[-1] if r2_frontier else {}
    tradeoff = ""
    if recommended_r2 is not None and r2_peak and r2_peak.get("id") != recommended_id:
        tradeoff = (
            f" The recommended single model ({recommended_id}) gives up a little Count R² — {recommended_r2:.3f} "
            f"versus {r2_peak['value']:.3f} from the earlier two-branch count blend ({r2_peak['id']}) — to be one "
            "model with the best mPQ+, so it sits below the R² peak on the left panel."
        )
    return f"""
      <h2>How we got here</h2>
      <p class='method-note'>Each dot is one recipe scored on the retrospective internal test; the line is the
      running best, which only steps when an attempt beats everything before it. {scored} recipes have been scored
      on this set across {lead_changes} lead changes. The broad result is more useful than any tiny endpoint:
      post-processing and calibration variants produced incremental or unstable gains, while switching to HoVer-Net,
      increasing rare-class exposure, and adding spatial TTA produced the recommended model
      (ringed).{tradeoff} The more complex two-model blend is reported in the table but excluded from the frontier.</p>
      <div class='traj-grid-wrap'>
        {_trajectory_panel('r2', 'Count R²', 'running best over all experiments', performance.get('trajectory', {}), show_target=False, recommended_id=recommended_id)}
        {_trajectory_panel('mpq', 'mPQ+', 'running best over all experiments', performance.get('trajectory', {}), show_target=False, recommended_id=recommended_id)}
      </div>
      <div class='traj-legend'>
        <span><i class='swatch-dot'></i>a scored recipe</span>
        <span><i class='swatch-lead'></i>took the lead</span>
        <span><i class='swatch-line'></i>running best</span>
        <span><i class='swatch-rec'></i>recommended model</span>
      </div>
    """


def _performance_html(performance: dict | None) -> tuple[str, str]:
    if not performance:
        return "<p>No aggregate experiment results were supplied.</p>", ""
    baseline = performance.get("baseline", {})
    best = performance.get("best", {})
    best_r2_row = performance.get("best_r2", {})
    best_mpq_row = performance.get("best_mpq", {})
    targets = performance.get("targets", {})

    def target_card(label: str, target, row: dict, key: str) -> str:
        value = row.get(key)
        if target is None or value is None:
            status, status_class, detail = "Pending", "pending", "No comparable completed result"
        else:
            gap = float(value) - float(target)
            status, status_class = ("Cleared", "passed") if gap >= 0 else ("Open", "open")
            detail = f"{gap:+.4f} versus target"
        return (
            f"<article class='target-card {status_class}'><div><span>{html.escape(label)} target</span>"
            f"<strong>{_metric_text(target)}</strong></div><div><span>Recommended model</span>"
            f"<strong>{_metric_text(value)}</strong></div><p><b>{status}</b> · {html.escape(detail)}</p></article>"
        )

    target_html = "".join(
        [
            target_card("Count R²", targets.get("r2"), best_r2_row, "r2"),
            target_card("mPQ+", targets.get("mpq"), best_mpq_row, "mpq"),
        ]
    )
    final_component = next((row for row in performance.get("rows", []) if row.get("id") == "E48"), {})
    final_status = html.escape(final_component.get("status", "queued"))
    final_stack_html = f"""
      <section class='runway' aria-label='What produced the improvement'>
        <div class='runway-head'><div><span class='eyebrow'>Improvement over baseline is driven by 4 changes</span><h2>{html.escape(best_mpq_row.get('method', 'Recommended final model'))}</h2></div></div>
        <ol class='runway-steps'>
          <li><b>1</b><div><strong>Switch from CellViT++ to HoVer-Net</strong><span>Train the joint cell-finding, separation, and typing architecture from generic ImageNet initialization.</span></div></li>
          <li><b>2</b><div><strong>Show it more rare-class patches</strong><span>Half of the sampling budget favors patches containing underrepresented cell types; the loss itself remains ordinary.</span></div></li>
          <li><b>3</b><div><strong>Average six spatial views at inference</strong><span>Align the original, flipped, and rotated predictions before separating cells once.</span></div></li>
          <li><b>4</b><div><strong>Use standard HoVer-Net cell separation</strong><span>Run it at the image resolution for which it was designed, then return masks at the dataset resolution.</span></div></li>
        </ol>
        <p class='method-note'><strong>Why stop here?</strong> More complicated count calibration and two-model typing variants scored slightly higher on one metric, but the gains did not justify changing a one-checkpoint model that already clears both targets.</p>
      </section>
    """

    def score_card(label: str, row: dict, deltas: bool = False) -> str:
        delta = ""
        delta_r2 = None if row.get("r2") is None or baseline.get("r2") is None else row["r2"] - baseline["r2"]
        delta_mpq = None if row.get("mpq") is None or baseline.get("mpq") is None else row["mpq"] - baseline["mpq"]
        if deltas and (delta_r2 is not None or delta_mpq is not None):
            parts = []
            if delta_r2 is not None:
                parts.append(f"R² {delta_r2:+.4f}")
            if delta_mpq is not None:
                parts.append(f"mPQ+ {delta_mpq:+.4f}")
            delta = f"<p class='delta'>vs baseline: {' · '.join(parts)}</p>"
        chip = _verdict_chip(row)
        return (
            f"<div class='score-card'><div class='score-head'><span class='eyebrow'>{html.escape(label)}</span>{chip}</div>"
            f"<h2>{html.escape(row.get('method', 'pending'))} <span class='score-id'>{html.escape(row.get('id', ''))}</span></h2>"
            f"<p class='evaluation-set'>{_provenance_chip(row.get('evaluation_set', 'Evaluation set not recorded'))}</p>"
            f"<div class='score-metrics'><div class='metric-primary' title='Macro mean of six per-class patch-count R² values; higher is better and negative is possible.'>"
            f"<b>{_metric_text(row.get('r2'))}</b><span>Count R²</span></div>"
            f"<div class='pq-family' title='mPQ+ is the mean of class-specific PQ; each class PQ equals DQ × SQ.'>"
            f"<div class='metric-primary'><b>{_metric_text(row.get('mpq'))}</b><span>mPQ+</span></div>"
            f"<div class='pq-components'><div title='Mean typed detection quality across classes'><b>{_metric_text(row.get('dq'))}</b><span>mDQ+</span></div>"
            f"<div title='Mean segmentation quality among matched typed nuclei across classes'><b>{_metric_text(row.get('sq'))}</b><span>mSQ+</span></div></div></div></div>{delta}</div>"
        )

    rows = []
    for row in performance.get("rows", []):
        baseline_r2 = baseline.get("r2")
        baseline_mpq = baseline.get("mpq")
        kind = html.escape(row.get("kind", "planned"))
        comparable = row.get("kind") != "benchmark" and row.get("delta_comparable", True)
        dr2 = None if not comparable or row.get("r2") is None or baseline_r2 is None else row["r2"] - baseline_r2
        dmpq = None if not comparable or row.get("mpq") is None or baseline_mpq is None else row["mpq"] - baseline_mpq
        status = html.escape(row.get("status", "planned"))
        rows.append(
            f"<tr class='result-{kind}'><td>{html.escape(row.get('stage', ''))}</td>"
            f"<td><strong>{html.escape(row.get('method', ''))}</strong><small>{html.escape(row.get('notes', ''))}</small></td>"
            f"<td><span class='status'>{status}</span></td><td>{_metric_text(row.get('r2'))}</td>"
            f"<td>{'—' if dr2 is None else f'{dr2:+.4f}'}</td><td>{_metric_text(row.get('mpq'))}</td>"
            f"<td>{'—' if dmpq is None else f'{dmpq:+.4f}'}</td>"
            f"<td>{_metric_text(row.get('dq'))}</td><td>{_metric_text(row.get('sq'))}</td>"
            f"<td>{html.escape(row.get('selection', ''))}</td></tr>"
        )
    references = []
    for reference in performance.get("published", {}).get("references", []):
        reported = []
        if reference.get("mpq_plus") is not None:
            reported.append(f"mPQ+ {float(reference['mpq_plus']):.4f}")
        if reference.get("r2") is not None:
            reported.append(f"R² {float(reference['r2']):.4f}")
        if not reported:
            continue
        source = html.escape(reference.get("source", ""), quote=True)
        references.append(
            f"<tr><td><a href='{source}' target='_blank' rel='noreferrer'>{html.escape(reference.get('name', ''))}</a></td>"
            f"<td>{html.escape(reference.get('split', ''))}</td><td>{' · '.join(reported)}</td>"
            f"<td>{html.escape(reference.get('role', ''))}</td></tr>"
        )
    note = html.escape(performance.get("published", {}).get("comparability_note", ""))
    rule = html.escape(performance.get("best_rule", ""))
    lesson_rows = "".join(
        f"<tr><td><span class='status'>{html.escape(item.get('priority', ''))}</span></td>"
        f"<td><strong>{html.escape(item.get('idea', ''))}</strong></td><td>{html.escape(item.get('evidence', ''))}</td></tr>"
        for item in performance.get("winner_lessons", [])
    )

    def ratio_text(value) -> str:
        return "—" if value is None else f"{float(value):.2f}×"

    shift_rows = "".join(
        f"<tr><td><strong>{html.escape(item.get('class', ''))}</strong></td>"
        f"<td>{100 * float(item.get('train_share', 0)):.2f}%</td>"
        f"<td>{100 * float(item.get('test_share', 0)):.2f}%</td>"
        f"<td>{float(item.get('actual_ratio', 0)):.2f}×</td>"
        f"<td>{ratio_text(item.get('em_ratio'))}</td></tr>"
        for item in performance.get("frequency_shift", [])
    )
    subgroup = performance.get("subgroups", {})

    validation_segmentation = performance.get("validation_segmentation", {})
    segmentation_html = ""
    if validation_segmentation:
        native = validation_segmentation.get("native", {})
        tta = validation_segmentation.get("tta", {})
        metric_specs = [
            ("R²", "r2"), ("mPQ+", "mpq"), ("mDQ+", "dq"), ("mSQ+", "sq"),
            ("foreground Jaccard", "foreground_jaccard"), ("bPQ", "bpq"),
            ("binary DQ", "binary_dq"), ("binary SQ", "binary_sq"),
            ("AJI+", "aji_plus"), ("boundary F1 (2 px)", "boundary_f1"),
        ]
        segmentation_rows = []
        for label, key in metric_specs:
            native_value, tta_value = native.get(key), tta.get(key)
            delta_value = None if native_value is None or tta_value is None else float(tta_value) - float(native_value)
            segmentation_rows.append(
                f"<tr><td><strong>{html.escape(label)}</strong></td><td>{_metric_text(native_value)}</td>"
                f"<td>{_metric_text(tta_value)}</td><td>{'—' if delta_value is None else f'{delta_value:+.4f}'}</td></tr>"
            )
        segmentation_html = (
            "<h2>Validation segmentation decomposition</h2>"
            "<p class='method-note'>mPQ+ is the mean of class-specific DQ×SQ values; mDQ+ and mSQ+ are "
            "reported separately, so their means do not generally multiply back to mPQ+. Binary metrics ignore "
            "cell type. Jaccard alone can look good despite merged or split nuclei, so bPQ, AJI+, and boundary F1 "
            "are retained as diagnostics. These metrics are not additional checkpoint-selection targets.</p>"
            "<div class='table-scroll'><table class='performance-table segmentation-table'><thead><tr>"
            "<th>metric</th><th>E32 ResNet-50 · single native view</th><th>E32 ResNet-50 · six-view spatial TTA</th><th>Δ TTA − native</th>"
            f"</tr></thead><tbody>{''.join(segmentation_rows)}</tbody></table></div>"
        )

    def subgroup_table(items: list[dict]) -> str:
        rendered = []
        for item in items:
            base_r2, best_r2 = item.get("baseline_r2"), item.get("best_r2")
            base_pq, best_pq = item.get("baseline_pq"), item.get("best_pq")
            delta_r2 = None if base_r2 is None or best_r2 is None or not np.isfinite([base_r2, best_r2]).all() else best_r2 - base_r2
            delta_pq = None if base_pq is None or best_pq is None or not np.isfinite([base_pq, best_pq]).all() else best_pq - base_pq
            samples = item.get("samples", {})
            support = f"({int(samples.get('patches', 0)):,}; {int(samples.get('count_gt', 0)):,}; {int(samples.get('mask_gt', 0)):,})"
            rendered.append(
                f"<tr><td><strong>{html.escape(item.get('label', ''))}</strong></td><td>{support}</td>"
                f"<td>{_metric_text(base_r2)}</td><td>{_metric_text(best_r2)}</td><td>{'—' if delta_r2 is None else f'{delta_r2:+.4f}'}</td>"
                f"<td>{_metric_text(base_pq)}</td><td>{_metric_text(best_pq)}</td><td>{'—' if delta_pq is None else f'{delta_pq:+.4f}'}</td>"
                f"<td>{_metric_text(item.get('baseline_dq'))}</td><td>{_metric_text(item.get('best_dq'))}</td>"
                f"<td>{_metric_text(item.get('baseline_sq'))}</td><td>{_metric_text(item.get('best_sq'))}</td></tr>"
            )
        return (
            "<div class='table-scroll'><table class='performance-table subgroup-table'><thead><tr>"
            "<th>group</th><th>support (patches; count GT; mask GT)</th><th>baseline R²</th><th>best R²</th><th>ΔR²</th>"
            "<th>baseline PQ</th><th>best PQ</th><th>ΔPQ</th><th>baseline DQ</th><th>best DQ</th>"
            "<th>baseline SQ</th><th>best SQ</th></tr></thead><tbody>" + "".join(rendered) + "</tbody></table></div>"
        )

    subgroup_html = ""
    if subgroup:
        scatter_payload = json.dumps(subgroup.get("scatter_points", []), separators=(",", ":")).replace("</", "<\\/")
        previous_option = (
            f"<option value='previous_pred'>{html.escape(subgroup['previous_name'])}</option>"
            if subgroup.get("previous_name")
            else ""
        )
        intermediate_option = (
            f"<option value='rotation_pred'>{html.escape(subgroup['intermediate_name'])}</option>"
            if subgroup.get("intermediate_name")
            else ""
        )
        scatter_html = f"""
          <h2>Count scatter underlying R²</h2>
          <p class='method-note'>Each point is one test patch × cell class. The displayed macro-R² is recomputed as the mean of finite per-class R² values, matching the leaderboard metric; selecting one class shows that class's R².</p>
          <div class='scatter-card'>
            <div class='scatter-controls'>
              <label>Method <select id='scatter-method'><option value='baseline_pred'>{html.escape(subgroup.get('baseline_name', 'Initial baseline'))}</option>{previous_option}{intermediate_option}<option value='best_pred' selected>{html.escape(subgroup.get('best_name', 'Current best'))}</option></select></label>
              <label>Class <select id='scatter-class'><option value='all'>All classes</option>{''.join(f"<option value='{html.escape(name)}'>{html.escape(name)}</option>" for name in CLASS_NAMES)}</select></label>
              <label>Color by <select id='scatter-color'><option value='class_name'>Cell class</option><option value='source'>Institution / source</option><option value='source_group'>Source image group</option><option value='gt_band'>GT-count band</option><option value='error_band'>Signed-error band</option></select></label>
              <strong id='scatter-r2'></strong>
            </div>
            <svg id='r2-scatter' viewBox='0 0 900 590' role='img' aria-label='Ground-truth versus predicted cell count scatter plot'></svg>
            <div id='scatter-legend' class='scatter-legend'></div>
          </div>
          <h2>Signed count error distributions (predicted − GT)</h2>
          <p class='method-note'>Negative errors are undercounts and positive errors are overcounts. Signed error is the primary view so directional bias remains visible; absolute magnitude is available only as a secondary toggle. Histograms default to percentages within each group so large sources do not dominate smaller ones. The institution × GT-count-band view tests density-dependent bias directly.</p>
          <div class='scatter-card error-card'>
            <div class='scatter-controls'>
              <label>Method <select id='count-error-method'><option value='baseline_pred'>{html.escape(subgroup.get('baseline_name', 'Initial baseline'))}</option>{previous_option}{intermediate_option}<option value='best_pred' selected>{html.escape(subgroup.get('best_name', 'Current best'))}</option></select></label>
              <label>Error <select id='count-error-mode'><option value='signed' selected>Signed error (predicted − GT)</option><option value='absolute'>Absolute magnitude</option></select></label>
              <label>Class <select id='count-error-class'><option value='all'>All classes</option>{''.join(f"<option value='{html.escape(name)}'>{html.escape(name)}</option>" for name in CLASS_NAMES)}</select></label>
              <label>Split by <select id='count-error-group'><option value='source' selected>Institution / source</option><option value='class_name'>Cell class</option><option value='source_group'>Source image group</option><option value='gt_band'>GT-count band</option><option value='source_gt_band'>Institution × GT-count band</option></select></label>
              <label>Y axis <select id='count-error-scale'><option value='percent' selected>Percent within group</option><option value='count'>Raw count</option></select></label>
              <label>Outlier threshold <select id='count-error-threshold'><option value='2'>|error| &gt; 2</option><option value='5' selected>|error| &gt; 5</option><option value='10'>|error| &gt; 10</option><option value='20'>|error| &gt; 20</option></select></label>
              <strong id='count-error-summary'></strong>
            </div>
            <svg id='count-error-hist' viewBox='0 0 1000 600' role='img' aria-label='Count error histograms split by metadata group'></svg>
            <div class='table-scroll'><table class='error-summary-table'><thead><tr><th>group</th><th>n</th><th>mean signed error</th><th>under</th><th>exact</th><th>over</th><th>MAE</th><th id='count-error-outlier-head'>outlier</th><th id='count-error-under-tail-head'>large under</th><th id='count-error-over-tail-head'>large over</th></tr></thead><tbody id='count-error-table'></tbody></table></div>
          </div>
          <script>(function(){{
            const points={scatter_payload};
            const svg=document.getElementById('r2-scatter'),method=document.getElementById('scatter-method'),classFilter=document.getElementById('scatter-class'),colorBy=document.getElementById('scatter-color'),label=document.getElementById('scatter-r2'),legend=document.getElementById('scatter-legend');
            const errorSvg=document.getElementById('count-error-hist'),errorMethod=document.getElementById('count-error-method'),errorMode=document.getElementById('count-error-mode'),errorClass=document.getElementById('count-error-class'),errorGroup=document.getElementById('count-error-group'),errorScale=document.getElementById('count-error-scale'),errorThreshold=document.getElementById('count-error-threshold'),errorSummary=document.getElementById('count-error-summary'),errorTable=document.getElementById('count-error-table'),outlierHead=document.getElementById('count-error-outlier-head'),underTailHead=document.getElementById('count-error-under-tail-head'),overTailHead=document.getElementById('count-error-over-tail-head');
            const ns='http://www.w3.org/2000/svg',palette=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf','#4c78a8','#f58518'];
            function add(name,attrs,text){{const el=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs||{{}}))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;svg.appendChild(el);return el}}
            function hash(s){{let h=0;for(let i=0;i<s.length;i++)h=(h*31+s.charCodeAt(i))>>>0;return h}}
            function band(value,cuts){{for(let i=0;i<cuts.length;i++)if(value<=cuts[i])return i;return cuts.length}}
            function category(p,pred){{if(colorBy.value==='gt_band')return ['0','1–2','3–5','6–10','>10'][band(p.gt,[0,2,5,10])];if(colorBy.value==='error_band'){{const e=pred-p.gt;return e<=-6?'undercount ≤−6':e<0?'undercount −5…−1':e===0?'exact':e<=5?'overcount +1…+5':'overcount ≥+6'}}return p[colorBy.value]}}
            function macroR2(rows,key){{const classes=[...new Set(rows.map(p=>p.class_name))],scores=[];for(const c of classes){{const x=rows.filter(p=>p.class_name===c),mean=x.reduce((s,p)=>s+p.gt,0)/x.length,sst=x.reduce((s,p)=>s+(p.gt-mean)**2,0),sse=x.reduce((s,p)=>s+(p[key]-p.gt)**2,0);if(sst>0)scores.push(1-sse/sst)}}return scores.length?scores.reduce((a,b)=>a+b,0)/scores.length:NaN}}
            function render(){{
              svg.replaceChildren();legend.replaceChildren();const key=method.value,rows=points.filter(p=>classFilter.value==='all'||p.class_name===classFilter.value),W=900,H=590,m={{l:70,r:25,t:25,b:62}},max=Math.max(1,...rows.flatMap(p=>[p.gt,p[key]])),limit=Math.ceil(max/10)*10,x=v=>m.l+v/max*(W-m.l-m.r),y=v=>H-m.b-v/max*(H-m.t-m.b);
              add('rect',{{x:m.l,y:m.t,width:W-m.l-m.r,height:H-m.t-m.b,fill:'#fff',stroke:'#ccd'}});for(let i=0;i<=5;i++){{const v=max*i/5;add('line',{{x1:x(v),y1:m.t,x2:x(v),y2:H-m.b,stroke:'#e5e7eb'}});add('line',{{x1:m.l,y1:y(v),x2:W-m.r,y2:y(v),stroke:'#e5e7eb'}});add('text',{{x:x(v),y:H-m.b+20,'text-anchor':'middle','font-size':12}},v.toFixed(0));add('text',{{x:m.l-10,y:y(v)+4,'text-anchor':'end','font-size':12}},v.toFixed(0))}}add('line',{{x1:x(0),y1:y(0),x2:x(max),y2:y(max),stroke:'#111','stroke-width':1.5,'stroke-dasharray':'6 4'}});add('text',{{x:(m.l+W-m.r)/2,y:H-16,'text-anchor':'middle','font-size':14,'font-weight':700}},'Ground-truth central-crop count');add('text',{{x:18,y:(m.t+H-m.b)/2,transform:`rotate(-90 18 ${{(m.t+H-m.b)/2}})`,'text-anchor':'middle','font-size':14,'font-weight':700}},'Predicted central-crop count');
              const categories=[...new Set(rows.map(p=>category(p,p[key])))].sort(),colors=new Map(categories.map((c,i)=>[c,palette[colorBy.value==='source_group'?hash(c)%palette.length:i%palette.length]]));for(const p of rows){{const pred=p[key],c=add('circle',{{cx:x(p.gt),cy:y(pred),r:2.8,fill:colors.get(category(p,pred)),'fill-opacity':.48,stroke:'none'}}),title=document.createElementNS(ns,'title');title.textContent=`patch ${{p.patch_id}} (${{p.patch_info}})\nclass: ${{p.class_name}}\ninstitution/source: ${{p.source}}\nsource group: ${{p.source_group}}\nGT: ${{p.gt}} · predicted: ${{pred}} · residual: ${{pred-p.gt>=0?'+':''}}${{pred-p.gt}}`;c.appendChild(title)}}
              const r2=macroR2(rows,key);label.textContent=`macro R² = ${{Number.isFinite(r2)?r2.toFixed(4):'—'}} · n=${{rows.length.toLocaleString()}} patch×class points`;for(const c of categories.slice(0,24)){{const item=document.createElement('span');item.innerHTML=`<i style="background:${{colors.get(c)}}"></i>${{c}}`;legend.appendChild(item)}}if(categories.length>24){{const item=document.createElement('span');item.textContent=`+ ${{categories.length-24}} more groups (hover points)`;legend.appendChild(item)}}
            }}
            function countBand(v){{return v===0?'0':v<=2?'1–2':v<=5?'3–5':v<=10?'6–10':v<=20?'11–20':'>20'}}
            function groupName(p){{if(errorGroup.value==='gt_band')return countBand(p.gt);if(errorGroup.value==='source_gt_band')return `${{p.source}} · GT ${{countBand(p.gt)}}`;return String(p[errorGroup.value]??'unknown')}}
            function errorBins(){{
              if(errorMode.value==='absolute')return [
                {{label:'0',test:v=>v===0}},{{label:'1–2',test:v=>v>=1&&v<=2}},{{label:'3–5',test:v=>v>=3&&v<=5}},{{label:'6–10',test:v=>v>=6&&v<=10}},{{label:'11–20',test:v=>v>=11&&v<=20}},{{label:'>20',test:v=>v>20}}
              ];
              return [
                {{label:'≤−21',test:v=>v<=-21}},{{label:'−20…−11',test:v=>v>=-20&&v<=-11}},{{label:'−10…−6',test:v=>v>=-10&&v<=-6}},{{label:'−5…−3',test:v=>v>=-5&&v<=-3}},{{label:'−2…−1',test:v=>v>=-2&&v<=-1}},{{label:'0',test:v=>v===0}},{{label:'1–2',test:v=>v>=1&&v<=2}},{{label:'3–5',test:v=>v>=3&&v<=5}},{{label:'6–10',test:v=>v>=6&&v<=10}},{{label:'11–20',test:v=>v>=11&&v<=20}},{{label:'≥21',test:v=>v>=21}}
              ];
            }}
            function renderErrors(){{
              errorSvg.replaceChildren();errorTable.replaceChildren();const key=errorMethod.value,rows=points.filter(p=>errorClass.value==='all'||p.class_name===errorClass.value),bins=errorBins(),by=new Map();
              for(const p of rows){{const name=groupName(p);if(!by.has(name))by.set(name,[]);by.get(name).push(p)}}
              let groups=[...by.entries()].sort((a,b)=>errorGroup.value==='source_group'?b[1].length-a[1].length:a[0].localeCompare(b[0],undefined,{{numeric:true}}));
              if(groups.length>24){{const shown=groups.slice(0,23),rest=groups.slice(23).flatMap(x=>x[1]);groups=[...shown,['Other groups',rest]]}}
              const stats=groups.map(([name,items])=>{{const residuals=items.map(p=>p[key]-p.gt),abs=residuals.map(Math.abs),counts=bins.map(b=>residuals.reduce((n,v)=>n+(b.test(errorMode.value==='absolute'?Math.abs(v):v)?1:0),0));return {{name,items,residuals,counts,n:items.length,mae:abs.reduce((a,b)=>a+b,0)/items.length,bias:residuals.reduce((a,b)=>a+b,0)/items.length,under:residuals.filter(v=>v<0).length,exact:residuals.filter(v=>v===0).length,over:residuals.filter(v=>v>0).length}}}});
              const threshold=Number(errorThreshold.value),overallResiduals=rows.map(p=>p[key]-p.gt),overallMae=overallResiduals.length?overallResiduals.reduce((s,v)=>s+Math.abs(v),0)/overallResiduals.length:NaN,overallBias=overallResiduals.length?overallResiduals.reduce((s,v)=>s+v,0)/overallResiduals.length:NaN,overallOutliers=overallResiduals.filter(v=>Math.abs(v)>threshold).length,overallUnder=overallResiduals.filter(v=>v<0).length,overallOver=overallResiduals.filter(v=>v>0).length;
              errorSummary.textContent=`mean signed error ${{Number.isFinite(overallBias)?(overallBias>=0?'+':'')+overallBias.toFixed(2):'—'}} · under ${{rows.length?(100*overallUnder/rows.length).toFixed(1):'—'}}% · over ${{rows.length?(100*overallOver/rows.length).toFixed(1):'—'}}% · MAE ${{Number.isFinite(overallMae)?overallMae.toFixed(2):'—'}} · outliers ${{rows.length?(100*overallOutliers/rows.length).toFixed(1):'—'}}% · n=${{rows.length.toLocaleString()}}`;
              outlierHead.textContent=`outlier |e| > ${{threshold}}`;underTailHead.textContent=`under e < −${{threshold}}`;overTailHead.textContent=`over e > ${{threshold}}`;
              const rowH=72,top=34,bottom=70,left=190,right=22,W=1000,H=Math.max(220,top+rowH*stats.length+bottom),plotW=W-left-right;errorSvg.setAttribute('viewBox',`0 0 ${{W}} ${{H}}`);
              const maximum=Math.max(1,...stats.flatMap(s=>s.counts.map(v=>errorScale.value==='percent'?100*v/s.n:v)));
              function hadd(name,attrs,text){{const el=document.createElementNS(ns,name);for(const [k,v] of Object.entries(attrs||{{}}))el.setAttribute(k,v);if(text!==undefined)el.textContent=text;errorSvg.appendChild(el);return el}}
              hadd('text',{{x:left,y:18,'font-size':12,fill:'#555'}},errorScale.value==='percent'?'Percent of patch×class points within each group':'Patch×class points');
              stats.forEach((s,gi)=>{{const y0=top+gi*rowH,rowBottom=y0+52,color=palette[gi%palette.length];hadd('text',{{x:left-10,y:y0+29,'text-anchor':'end','font-size':12,'font-weight':700}},s.name);hadd('line',{{x1:left,y1:rowBottom,x2:W-right,y2:rowBottom,stroke:'#b8bec6'}});s.counts.forEach((count,bi)=>{{const value=errorScale.value==='percent'?100*count/s.n:count,bw=plotW/bins.length-5,bh=value/maximum*45,x=left+bi*plotW/bins.length+2.5,y=rowBottom-bh,rect=hadd('rect',{{x,y,width:bw,height:bh,fill:color,'fill-opacity':.82}}),title=document.createElementNS(ns,'title');title.textContent=`${{s.name}}\n${{errorMode.value==='absolute'?'absolute error':'signed residual'}}: ${{bins[bi].label}}\n${{count.toLocaleString()}} / ${{s.n.toLocaleString()}} (${{(100*count/s.n).toFixed(1)}}%)`;rect.appendChild(title)}})}});
              bins.forEach((b,i)=>hadd('text',{{x:left+(i+.5)*plotW/bins.length,y:H-35,'text-anchor':'middle','font-size':11}},b.label));hadd('text',{{x:left+plotW/2,y:H-10,'text-anchor':'middle','font-size':13,'font-weight':700}},errorMode.value==='absolute'?'Absolute count error':'Signed count residual (predicted − GT)');
              for(const s of [...stats].sort((a,b)=>b.n-a.n)){{const tr=document.createElement('tr'),pct=v=>`${{(100*v/s.n).toFixed(1)}}%`,outlier=s.residuals.filter(v=>Math.abs(v)>threshold).length,underTail=s.residuals.filter(v=>v < -threshold).length,overTail=s.residuals.filter(v=>v > threshold).length;tr.innerHTML=`<td><strong>${{s.name}}</strong></td><td>${{s.n.toLocaleString()}}</td><td class="${{s.bias<0?'bias-under':s.bias>0?'bias-over':''}}">${{s.bias>=0?'+':''}}${{s.bias.toFixed(2)}}</td><td>${{pct(s.under)}}</td><td>${{pct(s.exact)}}</td><td>${{pct(s.over)}}</td><td>${{s.mae.toFixed(2)}}</td><td><strong>${{pct(outlier)}}</strong></td><td>${{pct(underTail)}}</td><td>${{pct(overTail)}}</td>`;errorTable.appendChild(tr)}}
            }}
            [method,classFilter,colorBy].forEach(el=>el.addEventListener('change',render));render();
            [errorMethod,errorMode,errorClass,errorGroup,errorScale,errorThreshold].forEach(el=>el.addEventListener('change',renderErrors));renderErrors();
          }})();</script>
        """
        def confusion_table(confusion: dict) -> str:
            columns = confusion.get("columns", [])
            counts = confusion.get("counts", [])
            normalized = confusion.get("row_normalized", [])
            header = "".join(f"<th>{html.escape(name)}</th>" for name in columns)
            body = []
            for row_name, row_counts, row_rates in zip(confusion.get("rows", []), counts, normalized):
                cells = []
                for count, rate in zip(row_counts, row_rates):
                    shade = min(0.82, 0.08 + 0.74 * float(rate)) if count else 0.0
                    style = f"background:rgba(33,113,181,{shade:.3f});color:{'white' if shade > 0.48 else '#202124'}" if count else ""
                    cells.append(f"<td style='{style}' title='{int(count):,} cases; {100 * float(rate):.2f}% of row'><b>{int(count):,}</b><small>{100 * float(rate):.1f}%</small></td>")
                body.append(f"<tr><th>{html.escape(row_name)}</th>{''.join(cells)}</tr>")
            return (
                f"<figure class='confusion-card'><figcaption><strong>{html.escape(confusion.get('name', ''))}</strong></figcaption>"
                f"<div class='table-scroll'><table class='confusion-table'><thead><tr><th>GT \\ predicted</th>{header}</tr></thead>"
                f"<tbody>{''.join(body)}</tbody></table></div></figure>"
            )

        confusion_html = "".join(confusion_table(item) for item in subgroup.get("confusions", []))
        subgroup_html = f"""
          {scatter_html}
          <h2>Metric breakdowns</h2>
          <p class='method-note'><strong>Compared:</strong> {html.escape(subgroup.get('baseline_name', 'baseline'))} versus {html.escape(subgroup.get('best_name', 'best'))}. {html.escape(subgroup.get('support_note', ''))}</p>
          <details open><summary><strong>By cell class</strong></summary>{subgroup_table(subgroup.get('by_class', []))}</details>
          <details open><summary><strong>By institution / domain proxy</strong></summary>{subgroup_table(subgroup.get('by_institution', []))}</details>
          <details><summary><strong>By institution × cell class</strong></summary>{subgroup_table(subgroup.get('by_both', []))}</details>
          <h2>Detection-aware confusion matrices</h2>
          <p class='method-note'>Rows are ground truth and columns are predictions. Each cell shows raw count and row-normalized percentage. “Missed GT” captures detection failures; “spurious prediction” captures unmatched detections, so the matrix explains both classification and PQ detection quality.</p>
          <div class='confusion-grid'>{confusion_html}</div>
        """
    best_mpq_row = performance.get("best_mpq", best)
    overview = f"""
      <h2>Improvement over baseline</h2>
      <p class='method-note'>Baseline is the initial CellViT-SAM-H model; the selected model is
      {html.escape(best_mpq_row.get('method', 'the recommended HoVer-Net'))}. Both are scored on the retrospective
      internal test set (657 patches).</p>
      {_improvement_tiles(performance)}
      {final_stack_html}
      <div class='headline-callout'><span class='eyebrow'>What is left</span>
        <p class='headline-diagnosis'>{_mpq_diagnosis(performance)}</p></div>
    """
    diagnostics = subgroup_html or "<p class='muted'>Subgroup diagnostics were not materialized for this build.</p>"
    return overview, diagnostics


def _metrics_html(performance: dict | None) -> str:
    if not performance:
        return "<p>No metric metadata was supplied.</p>"
    definitions = [
        ("Count R²", "The official count metric: compute R² separately for each of the six cell classes across patches, then take the macro mean. It measures count agreement, not mask overlap; it can be negative."),
        ("mPQ+", "The official typed instance-segmentation metric. Per-class true positives, false positives, false negatives, and matched-mask IoU are pooled across the evaluation set before class PQ values are averaged."),
        ("DQ", "Detection Quality = TP / (TP + 0.5 FP + 0.5 FN). It falls when nuclei are missed, duplicated, spuriously detected, or assigned the wrong class."),
        ("SQ", "Segmentation Quality = mean IoU among matched nuclei. It describes mask shape after a valid typed match exists; in this dashboard a class with zero true-positive matches is assigned SQ 0, so macro mSQ+ can also jump when a class gains its first match."),
        ("mDQ+ / mSQ+", "Macro means of class-specific DQ and SQ, exposed as diagnostic components. Their product is not generally mPQ+ because mean(DQ) × mean(SQ) differs from mean(DQ × SQ)."),
        ("bPQ", "Binary panoptic quality: PQ after ignoring cell type. Comparing bPQ with mPQ+ helps separate instance-detection/shape failures from typing failures."),
        ("Foreground Jaccard / Dice", "Pixel-overlap measures after collapsing all nuclei to one foreground class. Useful for gross coverage, but insensitive to some merged and split-instance errors."),
        ("AJI+", "Aggregated Jaccard Index with one-to-one instance matching. It is more instance-aware than foreground overlap and penalizes unmatched nuclei."),
        ("Boundary F1 (2 px)", "Precision/recall harmonic mean for predicted versus ground-truth nuclear boundaries within a two-pixel tolerance."),
    ]
    definition_cards = "".join(
        f"<article class='definition-card'><h3>{html.escape(name)}</h3><p>{html.escape(description)}</p></article>"
        for name, description in definitions
    )
    validation = performance.get("validation_segmentation", {})
    decomposition = ""
    if validation:
        native, tta = validation.get("native", {}), validation.get("tta", {})
        specs = [
            ("Count R²", "r2", "Macro classwise patch-count agreement"),
            ("mPQ+", "mpq", "Official typed panoptic quality"),
            ("mDQ+", "dq", "Typed detection component"),
            ("mSQ+", "sq", "Typed matched-mask quality component"),
            ("Foreground Jaccard", "foreground_jaccard", "Binary foreground intersection over union"),
            ("bPQ", "bpq", "Type-agnostic panoptic quality"),
            ("Binary DQ", "binary_dq", "Type-agnostic detection quality"),
            ("Binary SQ", "binary_sq", "Type-agnostic matched-mask quality"),
            ("AJI+", "aji_plus", "One-to-one aggregated instance overlap"),
            ("Boundary F1 (2 px)", "boundary_f1", "Boundary agreement within two pixels"),
        ]
        body = []
        for label, key, tooltip in specs:
            native_value, tta_value = native.get(key), tta.get(key)
            delta = None if native_value is None or tta_value is None else float(tta_value) - float(native_value)
            body.append(
                f"<tr><th title='{html.escape(tooltip, quote=True)}'>{html.escape(label)}</th>"
                f"<td>{_metric_text(native_value)}</td><td>{_metric_text(tta_value)}</td>"
                f"<td>{'—' if delta is None else f'{delta:+.4f}'}</td></tr>"
            )
        decomposition = f"""
          <h2>Worked diagnostic comparison</h2>
          <p class='method-note'><strong>Set: 711-patch source-group-disjoint development validation.</strong> These values explain the E32 TTA mechanism and are not test-set results or extra checkpoint-selection targets. The gain is primarily DQ-led: spatial averaging improves whether a correctly typed instance is recovered more than it changes the shape of already-matched masks.</p>
          <div class='table-scroll'><table class='performance-table segmentation-table'><thead><tr>
          <th title='Metric definition; hover row labels for detail'>Metric</th>
          <th title='One untransformed forward pass'>E32 ResNet-50 · single native view</th>
          <th title='Identity, flips, and rotations inverse-aligned and averaged before one decoder pass'>E32 ResNet-50 · six-view spatial TTA</th>
          <th title='Six-view value minus single-view value'>Δ TTA − native</th>
          </tr></thead><tbody>{''.join(body)}</tbody></table></div>
        """
    return f"""
      <div class='callout'><strong>Two official targets, two distinct questions.</strong> Count R² asks whether each patch contains the right number of each cell type. mPQ+ asks whether individual nuclei are detected, typed, and segmented correctly. We select recipes for these targets independently.</div>
      <div class='definition-grid'>{definition_cards}</div>
      <h2>Evaluation geometry</h2>
      <p class='method-note'>CoNIC inputs are 256×256 H&amp;E patches. Count R² uses nuclei whose centroids fall in the central 224×224 region, avoiding ambiguous edge counts. Instance masks are evaluated over the full patch according to the challenge contract. Every number elsewhere in this dashboard names its evaluation set because development-validation, internal-test, and published external-fold values are not interchangeable.</p>
      {decomposition}
    """


def _experiments_html(performance: dict | None) -> str:
    if not performance:
        return "<p>No experiment metadata was supplied.</p>"
    baseline = performance.get("baseline", {})
    baseline_r2, baseline_mpq = baseline.get("r2"), baseline.get("mpq")

    def delta(row: dict, key: str, baseline_value):
        if (
            row.get("kind") == "benchmark"
            or not row.get("delta_comparable", True)
            or row.get(key) is None
            or baseline_value is None
        ):
            return None
        return float(row[key]) - float(baseline_value)

    def cell(value, numeric: float | None = None, signed: bool = False) -> str:
        shown = "—" if value is None else (f"{float(value):+.4f}" if signed else _metric_text(value))
        sort_value = "-inf" if numeric is None or not np.isfinite(float(numeric)) else f"{float(numeric):.10f}"
        return f"<td data-sort-value='{sort_value}'>{shown}</td>"

    rendered = []
    experiment_rows = sorted(
        performance.get("rows", []),
        key=lambda row: (
            delta(row, "r2", baseline_r2) is not None,
            delta(row, "r2", baseline_r2) if delta(row, "r2", baseline_r2) is not None else -np.inf,
        ),
        reverse=True,
    )
    for row in experiment_rows:
        dr2, dmpq = delta(row, "r2", baseline_r2), delta(row, "mpq", baseline_mpq)
        visuals = "".join(
            f"<a class='evidence-thumb' href='../{html.escape(item.get('path', ''), quote=True)}' title='{html.escape(item.get('label', 'Open visual evidence'), quote=True)}' target='_blank'>"
            f"<img src='../{html.escape(item.get('path', ''), quote=True)}' alt='{html.escape(item.get('label', 'Experiment evidence'), quote=True)}' loading='lazy' onerror=\"this.closest('a').hidden=true\"></a>"
            for item in row.get("visuals", [])
        ) or "<span class='muted'>—</span>"
        recipe = row.get("recipe") or row.get("explanation") or row.get("notes", "")
        findings = row.get("findings", "")
        findings_html = (
            f"<details class='findings'><summary>result narrative</summary><p>{html.escape(findings)}</p></details>"
            if findings
            else ""
        )
        method_cell = (
            f"<td data-sort-value='{html.escape(row.get('method', ''))}'>"
            f"<strong>{html.escape(row.get('method', ''))}</strong>"
            f"<small class='recipe'>{html.escape(recipe)}</small>{findings_html}</td>"
        )
        rendered.append(
            f"<tr class='outcome-{html.escape(row.get('outcome', 'measured'))}'>"
            f"<td data-sort-value='{html.escape(row.get('id', ''))}'><strong>{html.escape(row.get('id', ''))}</strong></td>"
            f"<td data-sort-value='{html.escape(row.get('stage', ''))}'>{html.escape(row.get('stage', ''))}</td>"
            + method_cell
            + f"<td data-sort-value='{html.escape(row.get('outcome', ''))}'>{_verdict_chip(row)}</td>"
            f"<td data-sort-value='{html.escape(row.get('evaluation_set', ''))}'>{_provenance_chip(row.get('evaluation_set', ''))}</td>"
            + cell(row.get("r2"), row.get("r2"))
            + cell(dr2, dr2, signed=True)
            + cell(row.get("mpq"), row.get("mpq"))
            + cell(dmpq, dmpq, signed=True)
            + cell(row.get("dq"), row.get("dq"))
            + cell(row.get("sq"), row.get("sq"))
            + f"<td class='evidence-cell'>{visuals}</td>"
            f"<td>{html.escape(row.get('selection', ''))}</td></tr>"
        )

    headers = [
        ("ID", "Experiment identifier"), ("Role", "Whether this is a baseline, isolated ablation, combination, control, or external benchmark"),
        ("Method & result", "Fixed recipe, with the result narrative behind a disclosure"), ("Verdict", "Promotion outcome: promoted, partly promoted, rejected, measured, running, planned, cancelled, or external control"),
        ("Evaluation set", "The exact split on which displayed numbers were computed"),
        ("R²", "Macro mean of six classwise patch-count R² values"), ("ΔR²", "R² minus the internal initial baseline; external benchmarks are not differenced"),
        ("mPQ+", "Official typed panoptic quality"), ("ΔmPQ+", "mPQ+ minus the internal initial baseline"),
        ("mDQ+", "Diagnostic typed detection component"), ("mSQ+", "Diagnostic typed matched-mask quality component"),
        ("Visual evidence", "Click a thumbnail to inspect training, augmentation, or inference evidence at full size"),
        ("Selection / guardrail", "How hyperparameters were selected and leakage or comparability constraints"),
    ]
    header_html = "".join(
        f"<th title='{html.escape(tip, quote=True)}'><button class='sort-button' data-column='{index}'>{html.escape(label)}<span aria-hidden='true'></span></button></th>"
        for index, (label, tip) in enumerate(headers)
    )
    tally = performance.get("outcome_tally", [])
    tally_html = "".join(
        f"<div class='tally-chip'><span class='chip chip-{html.escape(item['outcome'], quote=True)}' title='{html.escape(item['hint'], quote=True)}'>{html.escape(item['label'])}</span><b>{item['count']}</b></div>"
        for item in tally
    )
    reference_rows = []
    for item in performance.get("idea_references", []):
        links = [f"<a href='{html.escape(item.get('source', ''), quote=True)}' target='_blank' rel='noreferrer'>primary source</a>"]
        if item.get("secondary_source"):
            links.append(f"<a href='{html.escape(item['secondary_source'], quote=True)}' target='_blank' rel='noreferrer'>secondary source</a>")
        reference_rows.append(
            f"<tr><td><strong>{html.escape(item.get('idea', ''))}</strong></td><td>{html.escape(item.get('origin', ''))}</td>"
            f"<td>{' · '.join(links)}</td><td>{html.escape(item.get('adaptation', ''))}</td><td>{html.escape(item.get('experiments', ''))}</td></tr>"
        )
    return f"""
      <div class='callout'><strong>Most of the work is negative results, and that is the point.</strong> We establish a
      leakage-controlled internal baseline, test one change at a time with learning rate and checkpoint selected on
      development validation, and only combine changes that survive in isolation. Each candidate is scored against an
      exact seed-, split-, and step-matched control. Only the changes marked <strong>Promoted</strong> are part of
      the selected final model; everything marked <strong>Not adopted</strong> was scored but left out — either it
      failed its control or a later change superseded it.</div>
      <div class='tally-strip'>{tally_html}</div>
      {_trajectory_html(performance)}
      <h2>Ablation and combination matrix</h2>
      <p class='method-note'>Default order is ΔR² descending; click any column heading to sort.
      Each row's verdict is a closed vocabulary (hover it for the meaning); the fixed recipe sits under the method name
      and the accumulated result narrative is behind “result narrative”. Blank metrics mean not yet run, not evaluated on
      that set, or not comparable. Provenance chips mark which split each number came from.</p>
      <div class='table-scroll'><table id='experiment-table' class='performance-table experiment-table'><thead><tr>{header_html}</tr></thead><tbody>{''.join(rendered)}</tbody></table></div>
      <h2>Idea attribution and adaptation</h2>
      <p class='method-note'>This project combines challenge definitions, published architectures, stain and TTA literature, the Pathology AI solution, and project-specific hypotheses. The table distinguishes an idea's origin from our implementation and validation guardrails.</p>
      <div class='table-scroll'><table class='performance-table reference-table'><thead><tr><th>idea</th><th>origin</th><th>references</th><th>our adaptation</th><th>experiments</th></tr></thead><tbody>{''.join(reference_rows)}</tbody></table></div>
    """


def _count_summary_html(case: dict) -> str:
    """Plain-language cell-count error summary for one patch.

    Leads with the total absolute count error (the sum of absolute per-class
    errors, which does not cancel), names the two largest gaps in words, then
    shows every class as an over/under chip.
    """
    errors = case.get("count_error", [])
    if not errors:
        return ""
    total = int(case.get("count_abs_error", sum(abs(int(value)) for value in errors)))
    pairs = sorted(zip(CLASS_NAMES, (int(value) for value in errors)), key=lambda pair: -abs(pair[1]))
    overs = [pair for pair in pairs if pair[1] > 0]
    unders = [pair for pair in pairs if pair[1] < 0]
    drivers = []
    if overs:
        drivers.append(f"{overs[0][0]} {overs[0][1]:+d} (too many)")
    if unders:
        drivers.append(f"{unders[0][0]} {unders[0][1]:+d} (too few)")
    lead = f"<strong>Total absolute count error across cell types: {total}.</strong>"
    if drivers:
        lead += " Biggest gaps: " + "; ".join(drivers) + "."
    chips = "".join(
        f"<span class='cc-chip cc-{'over' if value > 0 else 'under' if value < 0 else 'exact'}'>"
        f"{html.escape(name)} {value:+d}</span>"
        for name, value in pairs
    )
    return f"<div class='count-summary'><p>{lead}</p><div class='count-chips'>{chips}</div></div>"


def write_gallery(outdir: Path, cases: list[dict], triage: list[dict], performance: dict | None = None) -> None:
    gallery = outdir / "gallery"
    gallery.mkdir(parents=True, exist_ok=True)
    cards = []
    for case in cases:
        triage_row = next((x for x in triage if x["patch_id"] == case["patch_id"]), {})
        cards.append(f"""<article id='patch-{case['patch_id']}' class='card' data-error='{html.escape(triage_row.get('error_type',''))}'>
          <h3>Patch {case['patch_id']}</h3>
          <p class='case-meta'><span>{html.escape(case.get('source', 'unknown source'))}</span><span>{html.escape(case.get('patch_info', ''))}</span></p>
          {f"<p class='why-selected'><span>Why shown</span> {html.escape(case['selection_reason'])}</p>" if case.get('selection_reason') else ''}
          {_count_summary_html(case)}
          <a href='../{html.escape(case['panel'])}'><img class='panel' src='../{html.escape(case['panel'])}' title='Original patch, ground-truth cells, and predicted cells side by side, using shared cell-type colors' loading='lazy'></a>
          <h4>Individual nucleus cutouts</h4><p class='hint'>Every ground-truth nucleus in the patch, enlarged. A green border means the model found it and gave it the right cell type; a yellow border means it was missed, poorly outlined, or mistyped. Click for full size.</p>
          <div class='cutout-scroll'><a href='../{html.escape(case['cutouts'])}'><img class='cutouts' src='../{html.escape(case['cutouts'])}' loading='lazy'></a></div>
          <details><summary>Overlay animation</summary><img class='animation' src='../{html.escape(case['animation'])}' title='Cycles through the original patch, the ground-truth overlay, the prediction overlay, and a comparison overlay' loading='lazy'></details>
          <label>Human verdict <input data-patch='{case['patch_id']}' value='{html.escape(case['human_verdict'])}'></label>
          <label>Notes <textarea data-patch='{case['patch_id']}'>{html.escape(case['notes'])}</textarea></label>
        </article>""")
    class_legend = "".join(
        f"<span><i style='background:rgb({color[0]},{color[1]},{color[2]})'></i>{html.escape(name)}</span>"
        for index, name in enumerate(CLASS_NAMES, start=1)
        for color in [CLASS_COLORS[index]]
    )
    error_legend = "".join(
        f"<span><i style='background:rgb({color[0]},{color[1]},{color[2]})'></i>{html.escape(name)}</span>"
        for name, color in ERROR_COLORS.items()
    )
    overview_html, diagnostics_html = _performance_html(performance)
    metrics_html = _metrics_html(performance)
    experiments_html = _experiments_html(performance)
    page = """<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><meta name='color-scheme' content='light'><meta name='description' content='Interactive, leakage-aware CoNIC cell segmentation and count benchmark with experiment evidence and visual error review.'><meta property='og:type' content='website'><meta property='og:title' content='CoNIC experiment dashboard'><meta property='og:description' content='Compare cell segmentation and count methods, inspect subgroup metrics, and review ground truth against inferred nuclei.'><meta name='twitter:card' content='summary'><title>CoNIC experiment dashboard</title>
    <style>
    :root{--ink:#182230;--muted:#667085;--line:#dfe4ea;--surface:#fff;--canvas:#f5f7fa;--nav:#101828;--accent:#155eef;--green:#067647;--green-soft:#ecfdf3;--red:#b42318;--red-soft:#fef3f2;--shadow:0 1px 2px rgba(16,24,40,.05),0 4px 14px rgba(16,24,40,.06)}
    .target-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:.8rem;margin:1rem 0}.target-card{display:grid;grid-template-columns:1fr 1fr auto;gap:1rem;align-items:center;background:white;border:1px solid var(--line);border-left:4px solid #f79009;border-radius:10px;padding:.75rem 1rem;box-shadow:var(--shadow)}.target-card.passed{border-left-color:var(--green)}.target-card>div{display:flex;flex-direction:column}.target-card span{font-size:.7rem;color:var(--muted);font-weight:700;text-transform:uppercase;letter-spacing:.04em}.target-card strong{font-size:1.25rem;font-variant-numeric:tabular-nums}.target-card p{margin:0;white-space:nowrap;font-size:.8rem}.target-card.passed p b{color:var(--green)}.target-card.open p b{color:#b54708}.runway{background:linear-gradient(135deg,#101828,#1d2939);color:white;border-radius:12px;padding:1rem 1.1rem;box-shadow:var(--shadow)}.runway-head{display:flex;align-items:center;justify-content:space-between;gap:1rem}.runway h2{font-size:1.05rem;margin:.15rem 0}.runway .eyebrow{color:#98a2b3}.live-status{background:#fff3;border:1px solid #ffffff40;border-radius:999px;padding:.25rem .6rem;font-size:.72rem;white-space:nowrap}.runway-steps{list-style:none;padding:0;margin:.85rem 0 0;display:grid;grid-template-columns:repeat(5,1fr);gap:.6rem}.runway-steps li{display:flex;gap:.55rem;align-items:flex-start;border:1px solid #ffffff24;border-radius:8px;padding:.65rem;background:#ffffff09}.runway-steps li.active{border-color:#84adff;background:#155eef33}.runway-steps li>b{display:grid;place-items:center;min-width:1.45rem;height:1.45rem;border-radius:50%%;background:#ffffff20}.runway-steps li.active>b{background:#84adff;color:#102a56}.runway-steps div{display:flex;flex-direction:column;gap:.12rem}.runway-steps strong{font-size:.78rem}.runway-steps span{font-size:.68rem;color:#d0d5dd}@media(max-width:900px){.target-grid{grid-template-columns:1fr}.target-card{grid-template-columns:1fr 1fr}.target-card p{grid-column:1/-1}.runway-steps{grid-template-columns:1fr 1fr}}@media(max-width:560px){.runway-head{align-items:flex-start;flex-direction:column}.runway-steps{grid-template-columns:1fr}}
    *{box-sizing:border-box}html{scroll-behavior:smooth}body{font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;background:var(--canvas);color:var(--ink);line-height:1.45}.shell{max-width:1680px;margin:auto;padding:1.25rem 1.25rem 3rem}.topbar{position:sticky;top:0;z-index:20;background:rgba(16,24,40,.97);color:white;padding:.85rem max(1.25rem,calc((100vw - 1680px)/2 + 1.25rem));box-shadow:0 2px 12px rgba(0,0,0,.18);backdrop-filter:blur(10px)}.brand{display:flex;align-items:baseline;gap:.65rem;flex-wrap:wrap}.brand h1{font-size:1.05rem;margin:0}.brand span{color:#98a2b3;font-size:.78rem}.repo-link{margin-left:auto;color:#e4e7ec;text-decoration:none;border:1px solid #475467;border-radius:6px;padding:.3rem .55rem;font-size:.76rem;font-weight:700}.repo-link:hover{background:#344054;color:white}.tabs{display:flex;gap:.25rem;overflow-x:auto;margin-top:.7rem}.tabs button{border:0;background:transparent;color:#d0d5dd;border-radius:6px;padding:.48rem .72rem;font-weight:650;cursor:pointer;white-space:nowrap}.tabs button:hover{background:#1d2939;color:white}.tabs button.active{background:white;color:var(--nav)}.tab-panel{display:none}.tab-panel.active{display:block}.tab-panel>h1{font-size:1.55rem;margin:0 0 .25rem}.section-deck{color:var(--muted);margin:.1rem 0 1.2rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(620px,100%%),1fr));gap:1rem}.card,.scatter-card,.confusion-card,.curve-card,.definition-card,.callout,.legend{background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow)}.card{padding:1rem;min-width:0}.card img.panel,.card img.animation{width:100%%;height:auto;image-rendering:auto}.card h3{cursor:help;margin:.1rem 0}.case-meta{display:flex;gap:.4rem;flex-wrap:wrap;margin:.25rem 0}.case-meta span,.evaluation-set{display:inline-flex;background:#f2f4f7;border:1px solid #eaecf0;border-radius:999px;padding:.12rem .45rem;color:#475467;font-size:.72rem}.cutout-scroll,.table-scroll{overflow:auto}.cutout-scroll{background:#111827;padding:.35rem;border-radius:6px}.card img.cutouts{width:auto;max-width:none;height:auto;display:block}.card label{display:block;margin-top:.5rem}.card input,.card textarea{width:100%%;border:1px solid #d0d5dd;border-radius:6px;padding:.45rem}.card textarea{height:4rem}small,.hint,.muted{color:var(--muted)}.hint{font-size:.85rem;margin-top:-.25rem}.legend{padding:.7rem .85rem;margin:.6rem 0 1rem}.legend-row{display:flex;gap:.8rem;align-items:center;flex-wrap:wrap;margin:.25rem}.legend-row strong{min-width:7.4rem}.legend span{white-space:nowrap}.legend i{display:inline-block;width:.9rem;height:.9rem;margin-right:.3rem;border:1px solid #344054;vertical-align:-.1rem}h2{font-size:1.16rem;margin:1.8rem 0 .4rem}table{font-size:.75rem;border-collapse:separate;border-spacing:0;width:100%%;margin-top:.5rem;background:white}td,th{border-right:1px solid var(--line);border-bottom:1px solid var(--line);padding:.5rem;text-align:left;vertical-align:top}tr>*:first-child{border-left:1px solid var(--line)}thead th{position:sticky;top:0;z-index:2;background:#f2f4f7;border-top:1px solid var(--line);color:#344054}thead th:first-child{border-top-left-radius:7px}thead th:last-child{border-top-right-radius:7px}code{white-space:pre-wrap;word-break:break-word}.score-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:1rem;margin:1rem 0}.score-card{background:white;border:1px solid var(--line);border-radius:10px;padding:1rem 1.1rem;box-shadow:var(--shadow);border-top:4px solid #667085}.score-card:last-child{border-top-color:var(--green)}.score-card h2{font-size:1rem;margin:.25rem 0 .3rem;min-height:2.8em}.eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:.69rem;color:var(--muted);font-weight:800}.score-metrics{display:grid;grid-template-columns:1fr 1fr;gap:1rem;align-items:start;margin-top:.8rem}.metric-primary{display:flex;flex-direction:column}.metric-primary b{font-size:1.65rem;letter-spacing:-.03em;font-variant-numeric:tabular-nums}.metric-primary span{font-size:.8rem;color:var(--muted)}.pq-family>.metric-primary{padding-bottom:.45rem}.pq-components{display:grid;grid-template-columns:1fr 1fr;gap:.35rem;border-top:1px solid var(--line);padding-top:.4rem}.pq-components>div{display:flex;flex-direction:column;background:#f8fafc;border:1px solid var(--line);border-radius:5px;padding:.3rem .4rem}.pq-components b{font-size:.92rem;font-variant-numeric:tabular-nums}.pq-components span{font-size:.66rem;color:var(--muted)}.delta{color:var(--green);font-weight:700;font-size:.82rem}.method-note{max-width:1100px;color:#475467}.callout{border-left:4px solid var(--accent);padding:.8rem 1rem;margin:1rem 0;color:#344054}.definition-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.8rem}.definition-card{padding:.8rem .9rem}.definition-card h3{font-size:.95rem;margin:0 0 .25rem}.definition-card p{color:#475467;font-size:.84rem;margin:0}.performance-table{min-width:1000px;font-size:.79rem}.performance-table small{display:block;margin-top:.25rem;max-width:42rem}.performance-table td:nth-child(n+6):nth-child(-n+11){font-variant-numeric:tabular-nums;text-align:right}.segmentation-table{min-width:760px;max-width:1050px}.segmentation-table td:nth-child(n+2){font-variant-numeric:tabular-nums;text-align:right}.subgroup-table td:nth-child(n+3){font-variant-numeric:tabular-nums;text-align:right}.result-best,.result-combination{background:var(--green-soft)}.result-failed{background:var(--red-soft)}.result-baseline{background:#f8fafc}.status{white-space:nowrap;border-radius:999px;background:#eaecf0;padding:.15rem .45rem;font-size:.69rem}.experiment-table{min-width:1550px}.experiment-table th:nth-child(3){min-width:330px}.experiment-table th:nth-child(5){min-width:165px}.experiment-table th:last-child{min-width:250px}.sort-button{border:0;background:transparent;padding:0;color:inherit;font:inherit;font-weight:750;cursor:pointer;text-align:left}.sort-button span{padding-left:.22rem;color:var(--accent)}.sort-button[data-direction='asc'] span::after{content:'↑'}.sort-button[data-direction='desc'] span::after{content:'↓'}.evidence-cell{display:flex;gap:.35rem;min-width:145px;max-width:320px;flex-wrap:wrap}.evidence-thumb{display:block;border:1px solid var(--line);border-radius:5px;overflow:hidden;background:#fff}.evidence-thumb img{display:block;width:118px;height:72px;object-fit:contain}.reference-table{min-width:1100px}.reference-table th:nth-child(4){min-width:400px}details>summary{cursor:pointer;margin:.7rem 0 .25rem}.scatter-card{padding:1rem}.scatter-controls{display:flex;gap:1rem;align-items:end;flex-wrap:wrap}.scatter-controls label{display:flex;flex-direction:column;font-size:.78rem;font-weight:700}.scatter-controls select,.scatter-controls input{margin-top:.2rem;padding:.38rem;border:1px solid #d0d5dd;border-radius:5px;background:white}.scatter-controls strong{margin-left:auto}.scatter-card svg{width:100%%;max-width:1000px;height:auto}.scatter-legend{display:flex;gap:.6rem;flex-wrap:wrap;font-size:.75rem}.scatter-legend span{white-space:nowrap}.scatter-legend i{display:inline-block;width:.75rem;height:.75rem;margin-right:.25rem;border:1px solid #555;vertical-align:-.08rem}.error-summary-table{font-size:.78rem;max-width:1100px}.error-summary-table td:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums}.bias-under{color:var(--red);font-weight:700}.bias-over{color:#175cd3;font-weight:700}.confusion-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(620px,100%%),1fr));gap:1rem}.confusion-card{margin:.5rem 0;padding:.7rem}.confusion-table{min-width:760px}.confusion-table td{text-align:right;font-variant-numeric:tabular-nums}.confusion-table td small{display:block;color:inherit}.primary-button{border:0;border-radius:6px;background:var(--accent);color:white;padding:.55rem .8rem;font-weight:700;cursor:pointer}.dpath-note{border-left-color:#f79009}.dpath-links{display:flex;gap:.4rem;flex-wrap:wrap;margin-top:.45rem}.dpath-links a{background:#fffaeb;color:#93370d;border:1px solid #fedf89;border-radius:999px;padding:.18rem .5rem;text-decoration:none;font-size:.76rem}@media(max-width:720px){.shell{padding:.9rem}.score-metrics{grid-template-columns:1fr}.topbar{position:static}.tabs{padding-bottom:.2rem}.grid{grid-template-columns:1fr}.scatter-controls strong{margin-left:0;width:100%%}}
    /* ---- overhaul: system font, verdict chips, provenance, trajectory ---- */
    body{font-family:ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
    .headline-callout{background:linear-gradient(135deg,#f8fafc,#eef2ff);border:1px solid var(--line);border-left:4px solid var(--accent);border-radius:12px;padding:1rem 1.15rem;margin:1rem 0 1.25rem}
    .headline-claim{font-size:1.2rem;font-weight:750;letter-spacing:-.01em;margin:.2rem 0 .5rem;color:var(--ink)}
    .headline-diagnosis{margin:0;color:#344054;max-width:70ch}
    .chip{display:inline-flex;align-items:center;gap:.3rem;white-space:nowrap;border-radius:999px;padding:.16rem .55rem;font-size:.72rem;font-weight:700;border:1px solid transparent;line-height:1.3}
    .chip::before{content:'';width:.5rem;height:.5rem;border-radius:50%%;background:currentColor;flex:none}
    .chip-promoted{background:#ecfdf3;color:#067647;border-color:#abefc6}
    .chip-mixed{background:#fffaeb;color:#b54708;border-color:#fedf89}
    .chip-rejected{background:#fef3f2;color:#b42318;border-color:#fecdca}
    .chip-measured{background:#f2f4f7;color:#475467;border-color:#e4e7ec}
    .chip-running{background:#eff4ff;color:#155eef;border-color:#c7d7fe}
    .chip-planned{background:#f8fafc;color:#667085;border-color:#e4e7ec;border-style:dashed}
    .chip-cancelled{background:#f2f4f7;color:#98a2b3;border-color:#e4e7ec}
    .chip-baseline{background:#f2f4f7;color:#344054;border-color:#d0d5dd}
    .chip-external{background:#f4f3ff;color:#5925dc;border-color:#d9d6fe}
    .prov{display:inline-flex;align-items:center;white-space:nowrap;border-radius:5px;padding:.08rem .4rem;font-size:.68rem;font-weight:700;border:1px solid}
    .prov-test{background:#eff4ff;color:#155eef;border-color:#c7d7fe}
    .prov-val{background:#fffaeb;color:#b54708;border-color:#fedf89}
    .prov-external{background:#f4f3ff;color:#5925dc;border-color:#d9d6fe}
    .prov-none,.prov-other{background:#f2f4f7;color:#667085;border-color:#e4e7ec}
    .tally-strip{display:flex;flex-wrap:wrap;gap:.5rem;margin:1rem 0 .3rem}
    .tally-chip{display:inline-flex;align-items:center;gap:.4rem;background:white;border:1px solid var(--line);border-radius:999px;padding:.2rem .5rem .2rem .3rem;box-shadow:var(--shadow)}
    .tally-chip b{font-variant-numeric:tabular-nums;font-size:.95rem}
    .score-head{display:flex;align-items:center;justify-content:space-between;gap:.5rem}
    .score-card h2{min-height:0;display:flex;align-items:baseline;gap:.4rem;flex-wrap:wrap}
    .score-id{font-size:.72rem;font-weight:800;color:var(--accent);background:#eff4ff;border-radius:5px;padding:.05rem .35rem}
    .recipe{color:#667085}
    .findings{margin-top:.35rem}
    .findings>summary{font-size:.72rem;color:var(--accent);font-weight:700;margin:.1rem 0}
    .findings p{margin:.35rem 0 0;font-size:.75rem;color:#475467;max-width:60rem}
    tr.outcome-promoted{background:var(--green-soft)}
    tr.outcome-promoted td:first-child,tr.outcome-mixed td:first-child{box-shadow:inset 4px 0 0 var(--green)}
    tr.outcome-promoted td{font-weight:600}
    tr.outcome-mixed{background:#fffdf5}
    tr.outcome-rejected{background:var(--red-soft)}
    tr.outcome-baseline{background:#f8fafc}
    tr.outcome-external{background:#faf9ff}
    .experiment-table{min-width:1240px}
    .experiment-table th:nth-child(3){min-width:300px}.experiment-table th:nth-child(5){min-width:120px}
    .traj-grid-wrap{display:grid;grid-template-columns:1fr;gap:1rem;margin:.5rem 0}
    @media(min-width:1100px){.traj-grid-wrap{grid-template-columns:1fr 1fr}}
    .traj-panel{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:10px;box-shadow:var(--shadow);padding:.8rem .9rem}
    .traj-panel figcaption{display:flex;flex-direction:column;margin-bottom:.2rem}
    .traj-panel figcaption strong{font-size:.95rem}
    .traj-panel figcaption span{font-size:.74rem;color:var(--muted)}
    .traj-panel svg{width:100%%;height:auto}
    .traj-plot{fill:#fff;stroke:#e4e7ec}
    .traj-grid{stroke:#eef0f3;stroke-width:1}
    .traj-tick{fill:#898781;font-size:11px}
    .traj-target{stroke:#eda100;stroke-width:1.5;stroke-dasharray:5 4}
    .traj-target-label{fill:#b54708;font-size:11px;font-weight:700}
    .traj-dot{fill:#98a2b3;fill-opacity:.55}
    .traj-frontier{fill:none;stroke:#2a78d6;stroke-width:2.5;stroke-linejoin:round;stroke-linecap:round}
    .traj-lead{fill:#184f95;stroke:#fff;stroke-width:1.5}
    .traj-annotation{fill:#182230;font-size:11px;font-weight:750}
    .traj-recommended{fill:none;stroke:#067647;stroke-width:2.5}
    .traj-rec-label{fill:#067647;font-size:11px;font-weight:750}
    .swatch-rec{width:.75rem;height:.75rem;border-radius:50%%;border:2px solid #067647}
    .improvement-grid{display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin:1rem 0;align-items:start}
    .improve-tile{background:white;border:1px solid var(--line);border-top:4px solid var(--green);border-radius:12px;padding:1.1rem 1.25rem;box-shadow:var(--shadow)}
    .improve-pct{display:block;font-size:2.5rem;font-weight:800;letter-spacing:-.03em;line-height:1.02;margin-top:.15rem}
    .improve-pct.up,.improve-sub-metric b.up{color:var(--green)}
    .improve-pct.down,.improve-sub-metric b.down{color:var(--red)}
    .improve-sub{display:block;font-size:.8rem;color:var(--muted)}
    .improve-arrow{display:block;margin-top:.45rem;font-size:1.25rem;font-weight:750;font-variant-numeric:tabular-nums;color:var(--ink)}
    .improve-sub-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;margin-top:1rem;border-top:1px solid var(--line);padding-top:.8rem}
    .improve-sub-metric{display:flex;flex-direction:column;gap:.1rem}
    .improve-sub-metric>span:first-child{font-size:.68rem;color:var(--muted);font-weight:800;text-transform:uppercase;letter-spacing:.04em}
    .improve-sub-metric b{font-size:1.4rem;font-variant-numeric:tabular-nums;line-height:1.05}
    .improve-sub-arrow{font-size:.9rem;font-variant-numeric:tabular-nums;color:#475467}
    @media(max-width:720px){.improvement-grid{grid-template-columns:1fr}}
    .why-selected{margin:.25rem 0 .5rem;font-size:.8rem;color:#475467}
    .why-selected span{display:inline-block;font-size:.66rem;font-weight:800;text-transform:uppercase;letter-spacing:.04em;color:var(--accent);background:#eff4ff;border:1px solid #c7d7fe;border-radius:5px;padding:.05rem .35rem;margin-right:.35rem}
    .count-summary{margin:.5rem 0 .7rem}.count-summary p{margin:0 0 .45rem;font-size:.86rem;color:#344054}
    .count-chips{display:flex;flex-wrap:wrap;gap:.3rem}
    .cc-chip{font-size:.72rem;font-weight:700;border-radius:5px;padding:.12rem .42rem;font-variant-numeric:tabular-nums;border:1px solid;white-space:nowrap}
    .cc-over{background:#eff4ff;color:#175cd3;border-color:#c7d7fe}
    .cc-under{background:#fef3f2;color:#b42318;border-color:#fecdca}
    .cc-exact{background:#f2f4f7;color:#667085;border-color:#e4e7ec}
    .traj-legend{display:flex;flex-wrap:wrap;gap:1rem;font-size:.76rem;color:#475467;margin:.2rem 0 1rem}
    .traj-legend span{display:inline-flex;align-items:center;gap:.35rem}
    .traj-legend i{display:inline-block}
    .swatch-dot{width:.6rem;height:.6rem;border-radius:50%%;background:#98a2b3}
    .swatch-lead{width:.7rem;height:.7rem;border-radius:50%%;background:#184f95;border:1.5px solid #fff;box-shadow:0 0 0 1px #184f95}
    .swatch-line{width:1.1rem;height:0;border-top:2.5px solid #2a78d6}
    .swatch-target{width:1.1rem;height:0;border-top:2px dashed #eda100}
    </style></head><body>
    <header class='topbar'><div class='brand'><h1>CoNIC experiment dashboard</h1><a class='repo-link' href='https://github.com/cyrusmaher/cpath_conic' target='_blank' rel='noreferrer'>View code repository ↗</a></div><nav class='tabs' aria-label='Dashboard sections'><button class='active' data-tab='overview'>Where we stand</button><button data-tab='experiments'>What we tried</button><button data-tab='diagnostics'>Error analysis</button><button data-tab='metrics'>Metric definitions</button></nav></header><main class='shell'>
    <section class='callout' aria-labelledby='conic-background-title'>
      <span class='eyebrow'>Challenge background</span>
      <h2 id='conic-background-title'>What is CoNIC?</h2>
      <p><strong>CoNIC</strong> stands for <strong>Colon Nuclei Identification and Counting</strong>. Introduced for ISBI 2022, it benchmarks automated analysis of H&amp;E-stained colon histology: finding individual nuclei, outlining them, assigning one of six cell types—epithelial, lymphocyte, plasma, eosinophil, neutrophil, or connective tissue—and estimating the cellular composition of each image. The challenge dataset contains more than 535,000 annotated nuclei from 16 centres.</p>
      <p><strong>This dashboard reports a reproducible internal evaluation, not an official challenge submission.</strong> Its 657-patch locked test split is different from the challenge's hidden final test, so the numbers are not directly interchangeable with leaderboard scores.</p>
      <p><a class='primary-button' href='https://conic-challenge.grand-challenge.org/evaluation/segmentation-and-classification-final-test/leaderboard/' target='_blank' rel='noreferrer'>Official segmentation &amp; classification leaderboard ↗</a></p>
    </section>
    <section id='overview' class='tab-panel active'><h1>Where we stand</h1><p class='section-deck'>The recommended final model, how it is built, and its improvement over the initial baseline on both measures.</p>%s</section>
    <section id='experiments' class='tab-panel'><h1>What we tried</h1><p class='section-deck'>The path from baseline to the recommended model, then every ablation and combination with its verdict — mostly negative results, each scored against a matched control — plus complete idea attribution.</p>%s</section>
    <section id='diagnostics' class='tab-panel'><h1>Error analysis</h1><p class='section-deck'>Subgroup metrics, signed count bias, typed-detection confusion, and patch-level ground-truth review for the current best.</p>%s
    <h2>Patch-level review</h2><p class='method-note'>Ground truth versus the recommended rare-class-trained HoVer-Net model on the retrospective internal test set (657 patches). Every panel and GIF frame is labeled; ground truth and predictions share cell-class colors, comparison overlays use separate error-status colors.</p><div class='legend'><div class='legend-row'><strong>Cell classes</strong>%s</div><div class='legend-row'><strong>Comparison</strong>%s</div><div class='legend-row'><strong>Cutout border</strong><span><i style='background:rgb(34,197,94)'></i>correctly matched/classified</span><span><i style='background:rgb(250,204,21)'></i>error or missed</span></div></div><p>Inspect panels and animations, then record verdicts/notes. Download the edited review record before closing this page.</p><button class='primary-button' onclick='downloadReview()'>Download edited review JSON</button><div class='grid'>%s</div></section>
    <section id='metrics' class='tab-panel'><h1>Metric definitions</h1><p class='section-deck'>What each measure captures, how the components relate, and which evaluation set is being shown.</p>%s</section></main>
    <script>
    function showTab(id){const target=document.getElementById(id)?id:'overview';document.querySelectorAll('.tab-panel').forEach(x=>x.classList.toggle('active',x.id===target));document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x.dataset.tab===target));history.replaceState(null,'','#'+target);window.scrollTo({top:0})}
    function activate(id){document.querySelectorAll('.tab-panel').forEach(x=>x.classList.toggle('active',x.id===id));document.querySelectorAll('[data-tab]').forEach(x=>x.classList.toggle('active',x.dataset.tab===id))}
    document.querySelectorAll('[data-tab]').forEach(x=>x.addEventListener('click',()=>showTab(x.dataset.tab)));const initial=location.hash.slice(1);if(initial.startsWith('patch-'))activate('diagnostics');else if(initial)showTab(initial);window.addEventListener('hashchange',()=>{const id=location.hash.slice(1);if(id.startsWith('patch-'))activate('diagnostics')});
    function sortExperimentTable(column,button){const table=document.getElementById('experiment-table'),body=table.tBodies[0],rows=[...body.rows],previous=button.dataset.direction,direction=previous==='desc'?'asc':'desc';table.querySelectorAll('.sort-button').forEach(x=>delete x.dataset.direction);button.dataset.direction=direction;rows.sort((a,b)=>{const av=a.cells[column].dataset.sortValue??a.cells[column].textContent.trim(),bv=b.cells[column].dataset.sortValue??b.cells[column].textContent.trim(),an=Number(av),bn=Number(bv),numeric=!Number.isNaN(an)&&!Number.isNaN(bn);let result=numeric?an-bn:av.localeCompare(bv,undefined,{numeric:true});return direction==='asc'?result:-result});rows.forEach(row=>body.appendChild(row))}
    document.querySelectorAll('#experiment-table .sort-button').forEach(button=>button.addEventListener('click',()=>sortExperimentTable(Number(button.dataset.column),button)));const defaultSort=document.querySelector("#experiment-table .sort-button[data-column='6']");if(defaultSort)defaultSort.dataset.direction='desc';
    function downloadReview(){const rows=[...document.querySelectorAll('[data-patch]')];const by={};for(const el of rows){const id=el.dataset.patch;by[id]=by[id]||{patch_id:Number(id)};if(el.tagName==='TEXTAREA')by[id].notes=el.value;else by[id].human_verdict=el.value;}const blob=new Blob([JSON.stringify(Object.values(by),null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='review_records_edited.json';a.click();}
    </script></body></html>"""
    page = page % (overview_html, experiments_html, diagnostics_html, class_legend, error_legend, "\n".join(cards), metrics_html)
    (gallery / "index.html").write_text(page)
    (outdir / "index.html").write_text("""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="Interactive, leakage-aware CoNIC cell segmentation and count benchmark with experiment evidence and visual error review.">
  <meta property="og:type" content="website">
  <meta property="og:title" content="CoNIC experiment dashboard">
  <meta property="og:description" content="Compare cell segmentation and count methods, inspect subgroup metrics, and review ground truth against inferred nuclei.">
  <meta name="twitter:card" content="summary">
  <meta http-equiv="refresh" content="0; url=gallery/index.html">
  <title>CoNIC experiment dashboard</title>
</head>
<body>
  <p><a href="gallery/index.html">Open the CoNIC experiment dashboard</a></p>
</body>
</html>
""")
    (outdir / "review_records.json").write_text(json.dumps(cases, indent=2))
    with (outdir / "review_records.csv").open("w", newline="") as handle:
        fields = ["patch_id", "split", "human_verdict", "error_type", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: case.get(k, "") for k in fields} for case in cases)


def triage_cases(cases: Iterable[dict], max_cases: int = 40) -> list[dict]:
    rows = []
    for case in cases:
        error = float(case["count_abs_error"])
        if error >= 8:
            kind = "large_count_error"
            trigger = f"count_abs_error={error:.0f}"
        elif error >= 3:
            kind = "moderate_count_error"
            trigger = f"count_abs_error={error:.0f}"
        else:
            kind = "random_review"
            trigger = "stratified/random review"
        rows.append({"patch_id": case["patch_id"], "error_type": kind, "trigger": trigger, "confidence": 0.8 if error >= 8 else 0.5, "human_confirmed": False})
    rows.sort(key=lambda x: (-x["confidence"], x["patch_id"]))
    return rows[:max_cases]
