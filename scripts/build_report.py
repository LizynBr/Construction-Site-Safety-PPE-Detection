"""Collect measured experiment artifacts and write the report tables."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = ROOT / "reports"
    selection = json.loads((report / "selection.json").read_text())
    metrics = {}
    for key in ["e1-val", "e2-val", "e1-test", "e2-test"]:
        files = list((ROOT / "runs").rglob(key + "/metrics.json"))
        if len(files) != 1:
            raise RuntimeError(f"Missing or ambiguous metrics: {key}")
        src = files[0].parent
        dest = report / "metrics" / key
        dest.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.suffix in [".png", ".json", ".csv"]:
                shutil.copy2(p, dest / p.name)
        metrics[key] = json.loads(files[0].read_text())
    train = {}
    for name, folder in [("e1", "e1-opora"), ("e2", "e2-freeze10")]:
        src = ROOT / "runs/detect/runs/ppe" / folder
        dest = report / "training" / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in ["provenance.json", "args.yaml", "results.csv", "results.png"]:
            shutil.copy2(src / filename, dest / filename)
        train[name] = json.loads((src / "provenance.json").read_text())
    text = "# Лабораторная №3: обнаружение СИЗ на YOLOv8\n\n"
    text += "## Постановка и данные\n\nОбнаружение 10 классов СИЗ/людей/техники, YOLOv8n. Цель — воспроизводимое сравнение двух конфигураций; обязательного порога качества в задании нет. Набор 194 изображения: 135 train / 31 validation / 28 test. Происхождение, лицензия, прокси-группы сцен и ограничения описаны в [паспорте](../data/README.md). Test заморожен до обучения; выбор модели выполнен по validation, не по test.\n\n"
    text += "Исходный проект содержит результаты и веса, но не исполняемый код обучения. Исходный best.pt проверен на 5 изображениях, E0 (COCO YOLOv8n) — на 10. Общая COCO-модель обнаруживает людей и общие объекты, но не имеет отдельных PPE-классов; её классы нельзя напрямую сравнивать с PPE AP. Предсказания сохранены в upstream/ и e0/.\n\n"
    text += "## Конфигурации и результаты\n\nE1: 20 эпох, batch=4, imgsz=640, seed=42, AdamW lr0=.001, CPU. E2 меняет только freeze=10. Ресурсные изменения 50→20 эпох и 16→4 batch зафиксированы до сравнения. Гипотеза: заморозка backbone ускорит обучение ценой возможного ухудшения адаптации.\n\n"
    text += "|Опыт|Validation mAP50|Validation mAP50-95|Test mAP50|Test mAP50-95|Обучение, с|\n|---|---:|---:|---:|---:|---:|\n"
    for k in ["e1", "e2"]:
        v, t = metrics[k + "-val"], metrics[k + "-test"]
        text += f"|{k}|{v['map50']:.4f}|{v['map50_95']:.4f}|{t['map50']:.4f}|{t['map50_95']:.4f}|{train[k]['elapsed_seconds']:.1f}|\n"
    text += f"\nВыбрана **{selection['selected']}** по validation mAP50-95. [Фиксация выбора](selection.json). Начальные веса в обоих опытах одинаковы, upstream best.pt не использован.\n\n"
    for k in ["e1", "e2"]:
        text += f"### {k}: test по классам\n\n|Класс|Объекты|Precision|Recall|AP50|AP50-95|\n|---|---:|---:|---:|---:|---:|\n"
        for row in metrics[k + "-test"]["classes"]:
            text += f"|{row['class']}|{row['objects']}|{row['precision']:.4f}|{row['recall']:.4f}|{row['ap50']:.4f}|{row['ap50_95']:.4f}|\n"
        folder = report / "metrics" / f"{k}-test"
        for p in sorted(folder.glob("*.png")):
            if any(s in p.name.lower() for s in ["curve", "confusion"]):
                text += f"\n![{p.stem}](metrics/{k}-test/{p.name})\n"
    text += "\nPrecision/Recall в таблице Ultralytics соответствуют порогу около максимума среднего F1, а галереи используют фиксированный confidence=.25. Матрица Ultralytics: строки — предсказанный класс, столбцы — истинный; background-столбец — FP, background-строка — FN. AP усредняется по IoU от .50 до .95, mAP — по классам. При одном объекте AP machinery/vehicle не является устойчивой оценкой.\n\n"
    system = json.loads((report / "selected/system.json").read_text())
    text += "## Полная цепочка обработки\n\n"
    text += f"Batch=1, CPU, 3 прогревочных вызова исключены, {system['images']} кадров. Полный вызов включает декодирование, predict и сериализацию рамок; запись галерей исключена. p50={system['latency_p50_ms']:.2f} мс, p95={system['latency_p95_ms']:.2f} мс; {system['throughput_fps']:.2f} кадров/с; максимальный наблюдённый RSS={system['observed_peak_rss_mb']:.1f} MB.\n\n"
    text += "|Этап|Среднее, мс|\n|---|---:|\n"
    for k, v in system["stages_mean_ms"].items():
        text += f"|{k}|{v:.3f}|\n"
    text += f"\nНесопоставленных детектов на 100 кадров: {system['false_detections_per_100_frames']:.1f}; кадров хотя бы с одним FP на 100: {system['frames_with_false_detections_per_100']:.1f}. Это ошибки относительно исходной разметки, а не число производственных нарушений или тревог конкретной службы.\n\n"
    robust = json.loads((report / "selected/robustness.json").read_text())
    text += "|Условия|TP|FP|FN|Precision|Recall|\n|---|---:|---:|---:|---:|---:|\n"
    for k, r in robust.items():
        text += f"|{k}|{r['tp']}|{r['fp']}|{r['fn']}|{r['precision']:.3f}|{r['recall']:.3f}|\n"
    text += "\nСинтетические изменения освещения и перекрытия не заменяют независимую съёмку. Ракурс оценён качественно в аудите. [Политика отказа, drift и отката](system-policy.md).\n\n## Ошибки и ограничения\n\nШесть повторно создаваемых галерей находятся в selected/. Высокоуверенный FP без GT — кандидат для проверки разметки, а не доказанная ошибка эталона. Ручной разбор минимум 10 примеров: [анализ ошибок](error-analysis.md). Сопоставление диагностик — greedy по confidence и IoU, официальный AP рассчитывается Ultralytics отдельно.\n\n"
    text += "Ограничения: малый и неоднородный набор, редкие классы, неизвестные реальные ID камер, неоднозначные отрицательные PPE-метки и рамки; один seed не позволяет оценить дисперсию. Модель не допускается к автоматическим санкциям или управлению безопасностью. Следующий эксперимент — расширить независимые сцены и количество редких тестовых объектов, затем повторить опыт с несколькими seed.\n\n## Воспроизведение, проверки и ИИ\n\n[Команды, оборудование и использование ИИ](../README-lab03.md). [Аудит](annotation-audit.md), [проверка соответствия](compliance.md), [подготовка к защите](defense.md). Точные версии — requirements-lock.txt; конфигурации — configs/; commit и SHA256 весов/манифеста — training/*/provenance.json. Результаты получены локальными запусками. Внешняя рецензия, подтверждение студентом ручной проверки и защита требуют участия людей.\n"
    text += "\nМатрицы при confidence=.001 включают слабые гипотезы и не равны FP при рабочем .25. Ограничение трассировки E1 описано в experiment-plan.md. Разница на одном seed не доказывает общего преимущества заморозки.\n"
    improvement = report / "improvement-report.md"
    if improvement.exists():
        update = improvement.read_text(encoding="utf-8").replace(
            "# Дополнительный опыт: более длительное обучение",
            "## Обновление: E3, обучение 50 эпох (validation)",
            1,
        )
        update += "\n### E3: PR-кривая и зависимость от confidence на validation\n\n"
        for filename in (
            "BoxPR_curve.png",
            "BoxF1_curve.png",
            "BoxP_curve.png",
            "BoxR_curve.png",
            "confusion_matrix.png",
        ):
            update += f"![E3 validation — {filename}](improvement/validation/{filename})\n\n"
        update += "\n## Исходное сравнение E1/E2 и test-оценка\n\nСледующие таблицы, графики, системные измерения и галереи относятся к исходным E1/E2. E3 оценён только на validation; test-графики E1/E2 не являются графиками E3.\n\n"
        title, body = text.split("\n", 1)
        text = title + "\n\n" + update + body
    (report / "lab03-report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
