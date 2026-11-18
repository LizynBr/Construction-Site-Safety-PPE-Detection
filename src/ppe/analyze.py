"""Generate diagnostic galleries and CPU end-to-end measurements for a frozen model."""

import argparse
import json
import time

import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageEnhance

from .cli import environment, setup
from .data import ROOT, read_labels
from .vision import draw_boxes, match, serialize_prediction, xywh_to_xyxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--name", default="selected")
    args = parser.parse_args()
    setup()
    from ultralytics import YOLO

    model = YOLO(args.weights)
    records = [r for r in json.loads((ROOT / "data/manifest.json").read_text()) if r["split"] == args.split]
    out = ROOT / "reports" / args.name
    out.mkdir(parents=True, exist_ok=True)
    categories = [
        "high-confidence-fp",
        "critical-fn",
        "low-iou",
        "class-confusion",
        "small-occluded",
        "suspected-label",
    ]
    galleries = {k: [] for k in categories}
    measurements, diagnostics = [], []
    # Warm-up omitted from latency distribution.
    for _ in range(3):
        model.predict(str(ROOT / records[0]["image"]), device="cpu", verbose=False)
    peak = 0
    for index, row in enumerate(records):
        t0 = time.perf_counter()
        im = Image.open(ROOT / row["image"]).convert("RGB")
        im.load()
        t1 = time.perf_counter()
        result = model.predict(im, device="cpu", conf=0.25, iou=0.7, imgsz=640, verbose=False)[0]
        preds = serialize_prediction(result)
        t2 = time.perf_counter()
        gt = xywh_to_xyxy(read_labels(ROOT / row["label"]), *im.size)
        matched, fp, fn = match(gt, preds)
        any_matches, _, _ = match(gt, preds, threshold=0.1, class_aware=False)
        tags = []
        if any(preds[p]["confidence"] >= 0.5 for p in fp):
            tags.append(categories[0])
        if any(int(gt[g, 0]) in (0, 2, 7, 4) for g in fn):
            tags.append(categories[1])
        if any(
            0.1 <= overlap < 0.5 and int(gt[g, 0]) == preds[p]["class_id"] for g, p, overlap in any_matches
        ):
            tags.append(categories[2])
        if any(overlap >= 0.5 and int(gt[g, 0]) != preds[p]["class_id"] for g, p, overlap in any_matches):
            tags.append(categories[3])
        if any((g[3] - g[1]) * (g[4] - g[2]) < 0.0025 * im.width * im.height for g in gt):
            tags.append(categories[4])
        # Candidate only: high confidence unmatched detections may be model errors, never automatic relabels.
        if any(preds[p]["confidence"] >= 0.7 for p in fp):
            tags.append(categories[5])
        diagnostic = {
            "index": index,
            "image": row["image"],
            "original": row["original"],
            "predictions": preds,
            "fp": fp,
            "fn": fn,
            "matches": matched,
            "tags": tags,
        }
        diagnostics.append(diagnostic)
        annotated = draw_boxes(im, gt, preds)
        for tag in tags:
            directory = out / tag
            directory.mkdir(exist_ok=True)
            name = f"{index:03d}.jpg"
            annotated.save(directory / name)
            galleries[tag].append({"image": name, "original": row["original"], "fp": len(fp), "fn": len(fn)})
        peak = max(peak, psutil.Process().memory_info().rss)
        measurements.append(dict(decode_ms=(t1 - t0) * 1000, full_ms=(t2 - t0) * 1000, **result.speed))
    for tag, items in galleries.items():
        directory = out / tag
        directory.mkdir(exist_ok=True)
        text = (
            "# "
            + tag
            + "\n\nGreen: ground truth. Magenta: prediction. Diagnostic matching: IoU=.5, confidence=.25.\n\n"
        )
        text += (
            "\n\n".join(
                f"{x['original']} — FP={x['fp']}, FN={x['fn']}\n\n![boxes]({x['image']})" for x in items
            )
            or "No cases at the fixed threshold."
        )
        (directory / "README.md").write_text(text, encoding="utf-8")
    summary = environment()
    summary.update(
        weights=args.weights,
        split=args.split,
        images=len(records),
        latency_p50_ms=float(np.median([x["full_ms"] for x in measurements])),
        latency_p95_ms=float(np.percentile([x["full_ms"] for x in measurements], 95)),
        throughput_fps=1000 / float(np.mean([x["full_ms"] for x in measurements])),
        observed_peak_rss_mb=peak / 1024**2,
        false_detections_per_100_frames=100 * sum(len(x["fp"]) for x in diagnostics) / len(records),
        frames_with_false_detections_per_100=100 * sum(bool(x["fp"]) for x in diagnostics) / len(records),
        stages_mean_ms={k: float(np.mean([x[k] for x in measurements])) for k in measurements[0]},
        gallery_counts={k: len(v) for k, v in galleries.items()},
    )
    # Controlled perturbations retain geometry. Crop/rotation require label transforms and aren't claimed here.
    robustness = {}
    for label, transform in [
        ("normal", lambda x: x),
        ("dark", lambda x: ImageEnhance.Brightness(x).enhance(0.5)),
        ("bright", lambda x: ImageEnhance.Brightness(x).enhance(1.5)),
        ("occluded", occlude),
    ]:
        tp = fp_count = fn_count = 0
        for row in records:
            im = Image.open(ROOT / row["image"]).convert("RGB")
            gt = xywh_to_xyxy(read_labels(ROOT / row["label"]), *im.size)
            pred = serialize_prediction(
                model.predict(transform(im), device="cpu", conf=0.25, iou=0.7, verbose=False)[0]
            )
            m, f, n = match(gt, pred)
            tp += len(m)
            fp_count += len(f)
            fn_count += len(n)
        robustness[label] = {
            "tp": tp,
            "fp": fp_count,
            "fn": fn_count,
            "precision": tp / max(tp + fp_count, 1),
            "recall": tp / max(tp + fn_count, 1),
        }
    (out / "system.json").write_text(json.dumps(summary, indent=2))
    (out / "robustness.json").write_text(json.dumps(robustness, indent=2))
    (out / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (out / "timings.json").write_text(json.dumps(measurements, indent=2))
    print(json.dumps(summary, indent=2))


def occlude(image):
    im = image.copy()
    ImageDraw.Draw(im).rectangle(
        (im.width * 0.4, im.height * 0.35, im.width * 0.6, im.height * 0.65), fill="gray"
    )
    return im


if __name__ == "__main__":
    main()
