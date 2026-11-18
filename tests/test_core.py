import json

import numpy as np
import pytest
from PIL import Image

from ppe.data import ROOT, read_labels, validate_dataset, validate_manifest
from ppe.vision import draw_boxes, iou, match, xywh_to_xyxy


def test_experiments_change_only_freeze_factor():
    import yaml

    a = yaml.safe_load((ROOT / "configs/e1.yaml").read_text())
    b = yaml.safe_load((ROOT / "configs/e2.yaml").read_text())
    a.pop("name")
    b.pop("name")
    assert b.pop("freeze") == 10
    assert a == b


def test_wrong_class_is_not_a_true_positive():
    gt = np.array([[0, 0, 0, 20, 20]])
    predictions = [{"class_id": 2, "xyxy": [0, 0, 20, 20], "confidence": 0.9}]
    matched, fp, fn = match(gt, predictions)
    assert matched == [] and fp == [0] and fn == [0]


@pytest.mark.parametrize(
    "line", ["10 .5 .5 .2 .2", "0 1.1 .5 .2 .2", "0 .5 .5 0 .2", "0 nan .5 .2 .2", "0 .5"]
)
def test_invalid_labels(tmp_path, line):
    p = tmp_path / "bad.txt"
    p.write_text(line)
    with pytest.raises(ValueError):
        read_labels(p)


@pytest.mark.parametrize("field", ["group", "sha256", "name"])
def test_leak_detection(field):
    a = {"group": "a", "sha256": "a", "name": "a", "split": "train"}
    b = {"group": "b", "sha256": "b", "name": "b", "split": "test"}
    b[field] = a[field]
    with pytest.raises(ValueError, match="Leakage"):
        validate_manifest([a, b], check_files=False)


def test_frozen_dataset():
    counts = validate_dataset()
    assert all(min(c) > 0 for c in counts.values())
    assert len(json.loads((ROOT / "data/manifest.json").read_text())) >= 100


def test_visualization_and_coordinates():
    image = Image.new("RGB", (320, 240))
    boxes = xywh_to_xyxy([[0, 0.5, 0.5, 0.5, 0.5]], 320, 240)
    np.testing.assert_allclose(boxes[0, 1:], [80, 60, 240, 180])
    assert draw_boxes(image, boxes).size == image.size


def test_duplicate_detection_is_false_positive():
    gt = np.array([[0, 0, 0, 20, 20]])
    preds = [{"class_id": 0, "xyxy": [0, 0, 20, 20], "confidence": c} for c in [0.8, 0.9]]
    matches, fp, fn = match(gt, preds)
    assert len(matches) == 1 and fp == [0] and fn == []
    assert iou([0, 0, 20, 20], [10, 0, 30, 20]) == pytest.approx(1 / 3)


def test_control_prediction_schema():
    from ppe.cli import setup

    setup()
    from ultralytics import YOLO

    from ppe.vision import serialize_prediction

    row = json.loads((ROOT / "data/manifest.json").read_text())[0]
    result = YOLO(str(ROOT / "models/yolov8n.pt")).predict(
        str(ROOT / row["image"]), device="cpu", verbose=False
    )[0]
    objects = serialize_prediction(result)
    assert isinstance(objects, list)
    for obj in objects:
        assert 0 <= obj["confidence"] <= 1 and 0 <= obj["class_id"] < 80
        assert len(obj["xyxy"]) == 4 and all(np.isfinite(obj["xyxy"]))
