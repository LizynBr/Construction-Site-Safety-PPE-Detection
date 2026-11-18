"""Recreate the 40-image annotation review for the student from frozen files."""

import html
import json

from PIL import Image

from ppe.data import ROOT, read_labels
from ppe.vision import draw_boxes, xywh_to_xyxy


def main():
    selected = json.loads((ROOT / "reports/audit-selection.json").read_text())
    manifest = {r["image"]: r for r in json.loads((ROOT / "data/manifest.json").read_text())}
    out = ROOT / ".local/student-audit"
    out.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in selected:
        record = manifest[row["image"]]
        im = Image.open(ROOT / row["image"]).convert("RGB")
        gt = xywh_to_xyxy(read_labels(ROOT / record["label"]), *im.size)
        name = f"{row['id']:02d}.jpg"
        draw_boxes(im, gt).save(out / name)
        cards.append(
            f"<article><h2>{row['id']}: {html.escape(row['original'])}</h2>"
            f'<p>{row["type"]}</p><img src="{name}" width="640" height="640">'
            "<p>Проверить: пропуски, лишние рамки, класс, границы, перекрытия, дубли.</p></article>"
        )
    document = '<!doctype html><html lang="ru"><meta charset="utf-8"><title>Аудит разметки</title>'
    document += "<style>body{font:16px sans-serif;max-width:1000px;margin:24px auto}article{border-top:1px solid #bbb}img{max-width:100%;height:auto}</style>"
    document += "<h1>Проверка 30 случайных + 10 малых объектов</h1><p>Зелёные рамки: исходная разметка. Записать подтверждение и замечания в reports/annotation-audit.md.</p>"
    (out / "index.html").write_text(document + "".join(cards) + "</html>", encoding="utf-8")
    print(out / "index.html")


if __name__ == "__main__":
    main()
