"""Portable visualisation and explicit one-to-one diagnostic matching."""

import numpy as np
from PIL import ImageDraw

from .data import NAMES


def xywh_to_xyxy(labels, width, height):
    boxes = np.array(labels, dtype=float).reshape(-1, 5).copy()
    if len(boxes):
        x, y, w, h = boxes[:, 1:].T.copy()
        boxes[:, 1] = (x - w / 2) * width
        boxes[:, 2] = (y - h / 2) * height
        boxes[:, 3] = (x + w / 2) * width
        boxes[:, 4] = (y + h / 2) * height
    return boxes


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    tl = np.maximum(a[:2], b[:2])
    br = np.minimum(a[2:], b[2:])
    overlap = np.maximum(br - tl, 0).prod()
    area_a = np.maximum(a[2:] - a[:2], 0).prod()
    area_b = np.maximum(b[2:] - b[:2], 0).prod()
    return float(overlap / max(area_a + area_b - overlap, 1e-12))


def match(gt, predictions, threshold=0.5, class_aware=True):
    """Greedy confidence-order matching: duplicate detections stay false positives."""
    matched = []
    used = set()
    for p in sorted(range(len(predictions)), key=lambda i: -predictions[i]["confidence"]):
        pred = predictions[p]
        candidates = [
            (iou(box[1:], pred["xyxy"]), g)
            for g, box in enumerate(gt)
            if g not in used and (not class_aware or int(box[0]) == pred["class_id"])
        ]
        if candidates:
            overlap, g = max(candidates)
            if overlap >= threshold:
                used.add(g)
                matched.append((g, p, overlap))
    matched_p = {p for _, p, _ in matched}
    return (
        matched,
        [p for p in range(len(predictions)) if p not in matched_p],
        [g for g in range(len(gt)) if g not in used],
    )


def draw_boxes(image, gt=(), predictions=()):
    result = image.convert("RGB").copy()
    draw = ImageDraw.Draw(result)
    for row in gt:
        box = tuple(float(v) for v in row[1:])
        draw.rectangle(box, outline="#00dd44", width=2)
        draw.text((box[0], max(0, box[1] - 12)), "GT " + NAMES[int(row[0])], fill="#00dd44")
    for row in predictions:
        box = tuple(row["xyxy"])
        draw.rectangle(box, outline="#ff44bb", width=2)
        draw.text((box[0], box[1] + 2), f"P {row['class_name']} {row['confidence']:.2f}", fill="#ff44bb")
    return result


def serialize_prediction(result):
    predictions = []
    for box in result.boxes:
        cls = int(box.cls.item())
        predictions.append(
            {
                "class_id": cls,
                "class_name": result.names[cls],
                "confidence": float(box.conf.item()),
                "xyxy": box.xyxy[0].tolist(),
            }
        )
    return predictions
