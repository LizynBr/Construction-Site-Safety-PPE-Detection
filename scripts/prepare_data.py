"""Freeze a small source-grouped subset, or restore files from its committed manifest."""

import argparse
import json
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import imagehash
import numpy as np
import yaml
from PIL import Image, ImageDraw

from ppe.data import NAMES, ROOT, read_labels, sha256, validate_manifest


def source_group(name):
    name = name.split(".rf.")[0].lower()
    # Unknown video identities: deliberately combine all generic youtube and construction frames.
    if name.startswith("youtube-"):
        return "unknown-youtube-source"
    if name.startswith("construction-"):
        return "unknown-construction-source"
    if name.startswith("autox"):
        return "autox-recordings"
    if name.startswith("img_58"):
        return "img58-photo-session"
    video = re.split(r"_(?:mp4|mov)-", name)[0]
    if video != name:
        return "video-" + video
    return "image-" + name


def contact(records, out):
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(records), 20):
        canvas = Image.new("RGB", (1250, 1100), "white")
        draw = ImageDraw.Draw(canvas)
        for j, row in enumerate(records[start : start + 20]):
            im = Image.open(ROOT / row["source_image"]).convert("RGB").resize((250, 250))
            x, y = (j % 5) * 250, (j // 5) * 275
            canvas.paste(im, (x, y))
            draw.text((x + 3, y + 250), f"{start + j:03d} {row['original'][:29]}", fill="black")
        canvas.save(out / f"scenes-{start // 20:02d}.jpg")


def restore(records):
    for row in records:
        for kind in ("image", "label"):
            target = ROOT / row[kind]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / row["source_" + kind], target)
    print("Validated object counts:", validate_manifest(records))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    manifest = ROOT / "data/manifest.json"
    if manifest.exists():
        restore(json.loads(manifest.read_text()))
        return
    source = ROOT / "data/raw/source/css-data"
    originals = {}
    # Only unaugmented original validation/test images; old split is discarded completely.
    for folder in ("valid", "test"):
        for path in sorted((source / folder / "images").glob("*.jpg")):
            originals.setdefault(path.name.split(".rf.")[0], path)
    records = []
    for original, path in sorted(originals.items()):
        label = path.parent.parent / "labels" / (path.stem + ".txt")
        labels = read_labels(label)
        records.append(
            {
                "original": original,
                "name": path.name,
                "source_image": path.relative_to(ROOT).as_posix(),
                "source_label": label.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "label_sha256": sha256(label),
                "group": source_group(path.name),
                "classes": np.bincount(labels[:, 0].astype(int), minlength=10).tolist(),
            }
        )
    # Union explicit source groups and perceptually close scenes, before splitting.
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    hashes = [imagehash.phash(Image.open(ROOT / r["source_image"])) for r in records]
    for i, row in enumerate(records):
        for j in range(i):
            if (
                row["group"] == records[j]["group"]
                or row["sha256"] == records[j]["sha256"]
                or hashes[i] - hashes[j] <= 10
            ):
                parent[find(i)] = find(j)
    overrides = ROOT / "data/scene-overrides.json"
    if overrides.exists():
        for group in json.loads(overrides.read_text()):
            ids = [i for i, r in enumerate(records) if r["original"] in group]
            for i in ids[1:]:
                parent[find(i)] = find(ids[0])
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[find(i)].append(i)
    for members in groups.values():
        group_name = "scene-" + records[min(members)]["original"]
        for i in members:
            records[i]["group"] = group_name
    contact(records, ROOT / ".local/scenes")
    (ROOT / ".local/candidates.json").write_text(json.dumps(records, indent=2))
    if not args.freeze:
        print(f"{len(records)} candidates / {len(groups)} groups; inspect scene sheets before --freeze")
        return
    rng = random.Random(42)
    members = list(groups.values())
    sizes = np.array([len(g) for g in members])
    cls_counts = np.array([np.sum([records[i]["classes"] for i in g], axis=0) for g in members])
    best = None
    # Group-level random search using labels only; no model/test performance is consulted.
    for _ in range(30000):
        assignment = np.array(rng.choices(range(3), weights=[0.7, 0.15, 0.15], k=len(members)))
        counts = np.array([cls_counts[assignment == s].sum(axis=0) for s in range(3)])
        if np.any(counts == 0):
            continue
        numbers = np.array([sizes[assignment == s].sum() for s in range(3)])
        score = np.abs(numbers / len(records) - [0.7, 0.15, 0.15]).sum()
        score += 0.1 * np.abs(counts / counts.sum(axis=0) - np.array([0.7, 0.15, 0.15])[:, None]).mean()
        if best is None or score < best[0]:
            best = score, assignment.copy()
    if best is None:
        raise ValueError(
            "Cannot represent every class across independent groups; revise dataset scope explicitly"
        )
    for group, split_id in zip(members, best[1]):
        split = ("train", "val", "test")[split_id]
        for i in group:
            row = records[i]
            row.update(
                split=split,
                image=f"data/prepared/{split}/images/{row['name']}",
                label=f"data/prepared/{split}/labels/{Path(row['name']).stem}.txt",
            )
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    config = {
        "path": "data/prepared",
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": dict(enumerate(NAMES)),
    }
    (ROOT / "configs/ppe.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    restore(records)
    print("Manifest SHA256:", sha256(manifest))


if __name__ == "__main__":
    main()
