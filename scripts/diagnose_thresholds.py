"""Post-hoc diagnostic only: does not choose or change the fixed operating threshold."""

import json

from PIL import Image

from ppe.cli import setup
from ppe.data import ROOT, read_labels
from ppe.vision import match, serialize_prediction, xywh_to_xyxy


def main():
    setup()
    from ultralytics import YOLO

    selection = json.loads((ROOT / "reports/selection.json").read_text())
    model = YOLO(selection["weights"])
    rows = json.loads((ROOT / "reports/selected/diagnostics.json").read_text())
    manifest = {r["image"]: r for r in json.loads((ROOT / "data/manifest.json").read_text())}
    output = []
    for index in [4, 8, 11, 14, 19]:
        row = rows[index]
        im = Image.open(ROOT / row["image"]).convert("RGB")
        gt = xywh_to_xyxy(read_labels(ROOT / manifest[row["image"]]["label"]), *im.size)
        for conf, nms in [(0.25, 0.7), (0.1, 0.7), (0.25, 0.3), (0.25, 0.9)]:
            pred = serialize_prediction(model.predict(im, conf=conf, iou=nms, device="cpu", verbose=False)[0])
            tp, fp, fn = match(gt, pred)
            output.append(
                {
                    "index": index,
                    "confidence": conf,
                    "nms_iou": nms,
                    "tp": len(tp),
                    "fp": len(fp),
                    "fn": len(fn),
                }
            )
    (ROOT / "reports/selected/threshold-nms.json").write_text(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
