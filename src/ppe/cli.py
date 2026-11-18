"""Command line training, validation and predictions with recorded provenance."""

import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from .data import ROOT, sha256, validate_dataset

# Keep caches inside the workspace and do not enable telemetry integrations.
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".local/ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".local/matplotlib"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")


def setup():
    import numpy as np
    import torch

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(6)
    return torch


def environment():
    import importlib.metadata

    return {
        "python": platform.python_version(),
        "os": platform.platform(),
        "processor": platform.processor(),
        "packages": {
            p: importlib.metadata.version(p) for p in ("ultralytics", "torch", "torchvision", "numpy")
        },
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "manifest_sha256": sha256(ROOT / "data/manifest.json")
        if (ROOT / "data/manifest.json").exists()
        else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["validate-data", "train", "evaluate", "predict", "e0", "upstream"])
    parser.add_argument("--config", default="configs/e1.yaml")
    parser.add_argument("--weights", default="models/yolov8n.pt")
    parser.add_argument("--source")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--name", default="evaluation")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.action == "validate-data":
        print(json.dumps(validate_dataset(), indent=2))
        return
    setup()
    import yaml
    from ultralytics import YOLO

    from .vision import serialize_prediction

    start = time.perf_counter()
    meta = environment()
    if args.action == "train":
        validate_dataset()
        cfg = yaml.safe_load(Path(args.config).read_text())
        weights = cfg.pop("model")
        meta.update(config=cfg.copy(), initial_weights_sha256=sha256(weights))
        model = YOLO(weights)
        model.train(**cfg)
        folder = Path(model.trainer.save_dir)
        meta["elapsed_seconds"] = time.perf_counter() - start
        meta["best_weights_sha256"] = sha256(folder / "weights/best.pt")
        (folder / "provenance.json").write_text(json.dumps(meta, indent=2))
        print("RUN COMPLETE", folder, flush=True)
    elif args.action == "evaluate":
        validate_dataset()
        if (ROOT / "runs/eval" / args.name).exists():
            raise FileExistsError("Choose a new evaluation name to preserve the previous run")
        model = YOLO(args.weights)
        metrics = model.val(
            data=str(ROOT / "configs/ppe.yaml"),
            split=args.split,
            imgsz=args.imgsz,
            batch=1,
            device="cpu",
            workers=0,
            plots=True,
            conf=0.001,
            iou=0.7,
            project=str(ROOT / "runs/eval"),
            name=args.name,
            exist_ok=False,
        )
        counts = validate_dataset()[args.split]
        rows = []
        for j, cls in enumerate(metrics.box.ap_class_index):
            rows.append(
                {
                    "class_id": int(cls),
                    "class": model.names[int(cls)],
                    "objects": counts[int(cls)],
                    "precision": float(metrics.box.p[j]),
                    "recall": float(metrics.box.r[j]),
                    "ap50": float(metrics.box.ap50[j]),
                    "ap50_95": float(metrics.box.ap[j]),
                }
            )
        meta.update(
            split=args.split,
            weights_sha256=sha256(args.weights),
            imgsz=args.imgsz,
            map50=float(metrics.box.map50),
            map50_95=float(metrics.box.map),
            classes=rows,
            elapsed_seconds=time.perf_counter() - start,
        )
        # Current DetMetrics does not expose save_dir; project is absolute and name is fixed.
        folder = ROOT / "runs/eval" / args.name
        (folder / "metrics.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
    else:
        if args.action == "e0":
            records = json.loads((ROOT / "data/manifest.json").read_text())
            sources = [str(ROOT / r["image"]) for r in records if r["split"] == "train"][:10]
            weights = "models/yolov8n.pt"
            folder = ROOT / "reports/e0"
        elif args.action == "upstream":
            sources = [
                str(p)
                for p in (ROOT / "source_files").glob("*")
                if p.suffix in (".jpg", ".jfif") and p.stat().st_size
            ]
            weights = "models/best.pt"
            folder = ROOT / "reports/upstream"
        else:
            if not args.source:
                parser.error("--source is required for predict")
            sources, weights, folder = [args.source], args.weights, ROOT / "runs/predict" / args.name
        folder.mkdir(parents=True, exist_ok=True)
        model = YOLO(weights)
        output = []
        for source in sources:
            from PIL import Image

            result = model.predict(
                Image.open(source).convert("RGB"), imgsz=args.imgsz, device="cpu", conf=0.25, verbose=False
            )[0]
            result.save(filename=str(folder / Path(source).name))
            output.append({"image": source, "objects": serialize_prediction(result)})
        (folder / "predictions.json").write_text(json.dumps(output, indent=2))
        meta.update(weights_sha256=sha256(weights), elapsed_seconds=time.perf_counter() - start)
        (folder / "provenance.json").write_text(json.dumps(meta, indent=2))
        print("Saved", len(output), "predictions to", folder)


if __name__ == "__main__":
    main()
