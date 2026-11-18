"""Dataset validation and scene grouping; no model predictions are used to split data."""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

NAMES = [
    "Hardhat",
    "Mask",
    "NO-Hardhat",
    "NO-Mask",
    "NO-Safety Vest",
    "Person",
    "Safety Cone",
    "Safety Vest",
    "machinery",
    "vehicle",
]
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_labels(path, nc=10):
    rows = []
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if len(values) != 5:
            raise ValueError(f"{path}:{number}: expected class xc yc w h")
        cls, x, y, w, h = values
        if not all(np.isfinite(values)) or cls != int(cls) or not 0 <= cls < nc:
            raise ValueError(f"{path}:{number}: invalid class or nonfinite value")
        if not all(0 <= value <= 1 for value in (x, y, w, h)) or w <= 0 or h <= 0:
            raise ValueError(f"{path}:{number}: invalid normalized coordinates")
        rows.append(values)
    return np.array(rows, dtype=float).reshape(-1, 5)


def validate_manifest(records, check_files=True):
    ownership = {field: {} for field in ("group", "sha256", "name")}
    counts = {split: np.zeros(len(NAMES), dtype=int) for split in ("train", "val", "test")}
    for record in records:
        split = record["split"]
        if split not in counts:
            raise ValueError(f"Unknown split {split}")
        for field, owners in ownership.items():
            value = record[field]
            if value in owners and owners[value] != split:
                raise ValueError(f"Leakage: {field} {value}")
            owners[value] = split
        if check_files:
            image = ROOT / record["image"]
            if not image.is_file() or sha256(image) != record["sha256"]:
                raise ValueError(f"Missing/changed image {image}")
            label = ROOT / record["label"]
            if sha256(label) != record["label_sha256"]:
                raise ValueError(f"Changed label {label}")
            for cls in read_labels(label)[:, 0].astype(int):
                counts[split][cls] += 1
    return {split: values.tolist() for split, values in counts.items()}


def validate_dataset(config=ROOT / "configs/ppe.yaml"):
    cfg = yaml.safe_load(Path(config).read_text())
    base = Path(cfg["path"])
    if not base.is_absolute():
        base = ROOT / base
    records = json.loads((ROOT / "data/manifest.json").read_text())
    counts = validate_manifest(records)
    for split in ("train", "val", "test"):
        folder = base / cfg[split]
        expected = {Path(r["image"]).name for r in records if r["split"] == split}
        actual = {p.name for p in folder.glob("*.jpg")}
        if not expected or actual != expected:
            raise ValueError(f"YAML paths/files disagree with manifest: {split}")
    return counts
