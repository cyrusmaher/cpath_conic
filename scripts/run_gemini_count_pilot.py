#!/usr/bin/env python
"""Prepare and optionally submit a seven-image CoNIC counting audit to Gemini."""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cpath_conic.constants import CLASS_NAMES, COUNT_COLUMNS


def instance_boundaries(instances: np.ndarray) -> np.ndarray:
    boundaries = np.zeros_like(instances, dtype=bool)
    for axis in (0, 1):
        left = [slice(None), slice(None)]
        right = [slice(None), slice(None)]
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        changed = instances[tuple(left)] != instances[tuple(right)]
        positive = (instances[tuple(left)] > 0) | (instances[tuple(right)] > 0)
        edge = changed & positive
        boundaries[tuple(left)] |= edge
        boundaries[tuple(right)] |= edge
    return boundaries & (instances > 0)


def probability_rgb(values: np.ndarray, instances: np.ndarray) -> np.ndarray:
    """Fixed black→purple→orange→yellow heatmap."""
    anchors = np.asarray(
        [
            [0.00, 0, 0, 0],
            [0.15, 31, 12, 72],
            [0.40, 112, 31, 87],
            [0.70, 221, 89, 37],
            [1.00, 252, 253, 191],
        ],
        dtype=np.float32,
    )
    clipped = np.clip(values, 0.0, 1.0)
    output = np.zeros((*values.shape, 3), dtype=np.float32)
    for channel in range(3):
        output[..., channel] = np.interp(clipped, anchors[:, 0], anchors[:, channel + 1])
    output[instances == 0] = 0
    return output.astype(np.uint8)


def encode_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def resolve_project(explicit: str | None) -> str:
    if explicit:
        return explicit
    configured = subprocess.run(
        ["gcloud", "config", "get-value", "project"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if configured and configured != "(unset)":
        return configured
    adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    if adc.exists():
        quota_project = json.loads(adc.read_text()).get("quota_project_id")
        if quota_project:
            return str(quota_project)
    raise RuntimeError("No Google Cloud project found; pass --project")


def prepare(args: argparse.Namespace) -> tuple[list[Path], dict, str]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.prepared / "metadata.csv").sort_values("patch_id")
    row = metadata.loc[metadata.patch_id == args.patch_id]
    if len(row) != 1:
        raise ValueError(f"Patch {args.patch_id} not found exactly once")
    row = row.iloc[0]

    original = np.asarray(Image.open(args.prepared / "images" / f"{args.patch_id:05d}.png").convert("RGB"))
    predictions = np.load(args.predictions, mmap_mode="r")
    instances = np.asarray(predictions[args.patch_id, ..., 0], dtype=np.int32)
    probabilities = np.load(args.probabilities)
    selected = probabilities["patch_ids"] == args.patch_id
    instance_ids = probabilities["instance_ids"][selected].astype(np.int32)
    class_probs = probabilities["class_probs"][selected].astype(np.float32)
    probability_by_id = {int(instance_id): probs for instance_id, probs in zip(instance_ids, class_probs)}

    margin = args.margin
    crop = np.s_[margin : original.shape[0] - margin, margin : original.shape[1] - margin]
    original_crop = original[crop]
    instance_crop = instances[crop]
    size = (original_crop.shape[1] * args.upscale, original_crop.shape[0] * args.upscale)
    paths = []
    original_path = args.outdir / "00_original_he.png"
    Image.fromarray(original_crop).resize(size, Image.Resampling.LANCZOS).save(original_path)
    paths.append(original_path)

    for class_index, class_name in enumerate(CLASS_NAMES):
        values = np.zeros_like(instance_crop, dtype=np.float32)
        for instance_id in np.unique(instance_crop):
            if instance_id <= 0 or int(instance_id) not in probability_by_id:
                continue
            values[instance_crop == instance_id] = probability_by_id[int(instance_id)][class_index]
        rendered = probability_rgb(values, instance_crop)
        path = args.outdir / f"{class_index + 1:02d}_{class_name}_probability.png"
        rendered_up = np.asarray(Image.fromarray(rendered).resize(size, Image.Resampling.NEAREST)).copy()
        instance_up = np.asarray(
            Image.fromarray(instance_crop).resize(size, Image.Resampling.NEAREST),
            dtype=np.int32,
        )
        rendered_up[instance_boundaries(instance_up)] = np.asarray([0, 255, 255], dtype=np.uint8)
        Image.fromarray(rendered_up).save(path)
        paths.append(path)

    model_counts = np.load(args.counts)[args.patch_id].astype(int)
    gt_counts = row[COUNT_COLUMNS].to_numpy(dtype=int)
    proposal_count = int(len(np.unique(instance_crop[instance_crop > 0])))
    manifest = {
        "patch_id": args.patch_id,
        "source": str(row.source),
        "patch_info": str(row.patch_info),
        "images": [path.name for path in paths],
        "image_order": ["original H&E", *[f"{name} probability" for name in CLASS_NAMES]],
        "rendering": {
            "crop_margin": margin,
            "native_crop_shape": list(original_crop.shape),
            "upscale": args.upscale,
            "probability_scale": "fixed 0 to 1; black is zero, yellow is one; cyan is an instance boundary",
        },
        "sent_to_api": {
            "proposal_count": proposal_count if args.prompt_mode == "proposal-anchored" else None,
            "prompt_mode": args.prompt_mode,
        },
        "blinded_from_api": {
            "ground_truth_counts": dict(zip(CLASS_NAMES, gt_counts.tolist())),
            "model_counts": dict(zip(CLASS_NAMES, model_counts.tolist())),
        },
    }
    (args.outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    prompt = """You are auditing automated nuclear segmentation and cell typing in one H&E pathology crop.

You receive seven aligned images in this exact order: original H&E, then neutrophil, epithelial, lymphocyte, plasma, eosinophil, and connective-tissue probability maps. In every probability map, each proposed nucleus is filled with that nucleus's model probability for the named class. The color scale is fixed across maps: black is probability 0, yellow is probability 1, and thin cyan lines mark predicted instance boundaries.

The probability maps are fallible proposals, not ground truth. Count biological nuclei by class. Use the H&E image to add nuclei missed by all maps, remove artifacts, split merged proposals, and correct cell types by considering morphology and all six probabilities jointly. Count one biological nucleus exactly once. The images already show only the official central counting crop; partial nuclei visible at an image edge still count.

Return integer counts for all six classes, estimated numbers of four correction types, a confidence from 0 to 1, and a concise reasoning summary. Do not merely report connected-component counts and do not assume the highest-probability class is always correct."""
    if args.prompt_mode == "proposal-anchored":
        prompt += f"""

COUNT-CONSERVING AUDIT: The cyan boundaries encode exactly {proposal_count} distinct proposed nuclei, repeated consistently across all six probability maps. First assign every one of these {proposal_count} proposals to exactly one of the six classes; your six proposal_counts must sum to {proposal_count}. Do not omit a proposal because it is low-confidence or hard to see. Then inspect the H&E image for explicit corrections. Final total count must equal proposal total − false positives removed + missed nuclei added + additional nuclei created by splitting merged regions. Report proposal_counts separately from final counts. Use cells_retyped only to describe class reassignment; retyping does not change the total count."""
    (args.outdir / "prompt.txt").write_text(prompt)
    return paths, manifest, prompt


def submit(args: argparse.Namespace, paths: list[Path], prompt: str, manifest: dict) -> dict:
    project = resolve_project(args.project)
    token = subprocess.check_output(
        ["gcloud", "auth", "application-default", "print-access-token"],
        text=True,
    ).strip()
    labels = ["original H&E", *[f"{name} probability map" for name in CLASS_NAMES]]
    parts: list[dict] = [{"text": prompt}]
    for label, path in zip(labels, paths):
        parts.extend(
            [
                {"text": f"Next image: {label}."},
                {"inlineData": {"mimeType": "image/png", "data": encode_png(path)}},
            ]
        )
    count_properties = {name: {"type": "INTEGER", "minimum": 0} for name in CLASS_NAMES}
    correction_names = ["missed_nuclei_added", "false_positives_removed", "merged_regions_split", "cells_retyped"]
    schema = {
        "type": "OBJECT",
        "properties": {
            "counts": {"type": "OBJECT", "properties": count_properties, "required": CLASS_NAMES},
            "estimated_corrections": {
                "type": "OBJECT",
                "properties": {name: {"type": "INTEGER", "minimum": 0} for name in correction_names},
                "required": correction_names,
            },
            "confidence": {"type": "NUMBER", "minimum": 0, "maximum": 1},
            "reasoning_summary": {"type": "STRING"},
        },
        "required": ["counts", "estimated_corrections", "confidence", "reasoning_summary"],
    }
    if args.prompt_mode == "proposal-anchored":
        schema["properties"]["proposal_counts"] = {
            "type": "OBJECT",
            "properties": count_properties,
            "required": CLASS_NAMES,
        }
        schema["required"].insert(0, "proposal_counts")
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "seed": 20260715,
            "maxOutputTokens": 2048,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    endpoint = (
        f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/{args.location}/"
        f"publishers/google/models/{args.model}:generateContent"
    )
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            raw = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise RuntimeError(f"Vertex request failed ({exc.code}): {body[:2000]}") from exc
    (args.outdir / "vertex_response.json").write_text(json.dumps(raw, indent=2))
    text = raw["candidates"][0]["content"]["parts"][0]["text"]
    parsed = json.loads(text)
    result = {
        "model": args.model,
        "location": args.location,
        "response": parsed,
        "usage_metadata": raw.get("usageMetadata", {}),
        "model_version": raw.get("modelVersion"),
        "response_id": raw.get("responseId"),
        "prompt_mode": args.prompt_mode,
        "proposal_count": manifest.get("sent_to_api", {}).get("proposal_count"),
    }
    (args.outdir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--patch-id", type=int, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--margin", type=int, default=16)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--project")
    parser.add_argument("--location", default="global")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--prompt-mode", choices=["free", "proposal-anchored"], default="free")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    paths, manifest, prompt = prepare(args)
    print(json.dumps({"prepared": manifest, "outdir": str(args.outdir)}, indent=2))
    if args.submit:
        print(json.dumps(submit(args, paths, prompt, manifest), indent=2))


if __name__ == "__main__":
    main()
