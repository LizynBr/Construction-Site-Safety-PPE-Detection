"""Evaluate E3 on validation only and preserve the original laboratory report."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    training = ROOT / "runs/detect/runs/ppe/e3-freeze10-50epochs"
    if not (training / "provenance.json").exists():
        raise RuntimeError("E3 training has not completed")
    metrics_file = ROOT / "runs/eval/e3-val/metrics.json"
    if not metrics_file.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ppe.cli",
                "evaluate",
                "--weights",
                str(training / "weights/best.pt"),
                "--split",
                "val",
                "--name",
                "e3-val",
            ],
            cwd=ROOT,
            check=True,
        )
    new = json.loads(metrics_file.read_text())
    old = json.loads((ROOT / "reports/metrics/e2-val/metrics.json").read_text())
    out = ROOT / "reports/improvement"
    out.mkdir(parents=True, exist_ok=True)
    for source in [training, metrics_file.parent]:
        target = out / ("training" if source == training else "validation")
        target.mkdir(exist_ok=True)
        for p in source.iterdir():
            if p.suffix in (".json", ".csv", ".yaml", ".png"):
                shutil.copy2(p, target / p.name)
    delta = new["map50_95"] - old["map50_95"]
    comparison = {
        "baseline": "E2, 20 epochs",
        "candidate": "E3, 50 epochs",
        "baseline_map50_95": old["map50_95"],
        "candidate_map50_95": new["map50_95"],
        "delta": delta,
        "improved_validation": delta > 0,
        "test_reevaluated": False,
        "candidate_weights": str((training / "weights/best.pt").relative_to(ROOT)),
    }
    (out / "comparison.json").write_text(json.dumps(comparison, indent=2))
    text = "# Дополнительный опыт: более длительное обучение\n\n"
    text += "Изменено только число эпох: 20→50, остальные настройки E2 сохранены. План зафиксирован до запуска в improvement-plan.md. Набор и разбиение неизменны.\n\n"
    text += "|Опыт|Validation mAP50|Validation mAP50-95|\n|---|---:|---:|\n"
    for label, row in [("E2 (20 эпох)", old), ("E3 (50 эпох)", new)]:
        text += f"|{label}|{row['map50']:.4f}|{row['map50_95']:.4f}|\n"
    text += f"\nИзменение mAP50-95: {delta:+.4f} ({100 * delta:+.2f} процентного пункта). "
    text += (
        "Кандидат лучше по зафиксированному validation-критерию.\n\n"
        if delta > 0
        else "Гипотеза улучшения validation не подтверждена.\n\n"
    )
    text += "|Класс|Объекты val|AP50-95 E2|AP50-95 E3|Recall E2|Recall E3|\n|---|---:|---:|---:|---:|---:|\n"
    before = {r["class_id"]: r for r in old["classes"]}
    for r in new["classes"]:
        b = before[r["class_id"]]
        text += f"|{r['class']}|{r['objects']}|{b['ap50_95']:.4f}|{r['ap50_95']:.4f}|{b['recall']:.4f}|{r['recall']:.4f}|\n"
    text += "\n![Обучение](improvement/training/results.png)\n\n"
    text += "Исходный отчёт E1/E2 сохранён. Test повторно не оценивался: он уже просмотрен, поэтому для независимого подтверждения улучшения нужна новая тестовая выборка. Разница на одном seed и маленьком validation не является гарантией качества на других камерах. Precision/Recall библиотека считает при пороге, близком к максимуму среднего F1.\n\n"
    text += "Главное оставшееся ограничение — 135 неоднородных обучающих изображений и мало validation-объектов отдельных классов; увеличение эпох не заменяет новые независимые размеченные сцены. Плохие/неоднозначные рамки требуют проверки человеком.\n"
    text += "\nAP Hardhat снизился 0.0863→0.0625, AP NO-Safety Vest — 0.1565→0.1317. Прирост среднего результата обеспечивают прежде всего Mask и vehicle. E3 не объявляется готовой заменой для контроля касок; следующий приоритет — новые независимые строительные сцены и проверка разметки.\n"
    (ROOT / "reports/improvement-report.md").write_text(text, encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
