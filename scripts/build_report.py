"""Build the readable laboratory report from committed experimental evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def table(headers, rows):
    return (
        "| "
        + " | ".join(headers)
        + " |\n|"
        + "|".join(["---"] * len(headers))
        + "|\n"
        + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
    )


def main():
    metrics = {
        key: read(f"reports/metrics/{key}/metrics.json") for key in ("e1-val", "e2-val", "e1-test", "e2-test")
    }
    e3 = read("reports/improvement/validation/metrics.json")
    manifests = read("data/manifest.json")
    counts = {
        split: [sum(r["classes"][i] for r in manifests if r["split"] == split) for i in range(10)]
        for split in ("train", "val", "test")
    }
    names = [row["class"] for row in e3["classes"]]
    replacements = {}
    replacements["CLASS_COUNTS"] = table(
        ["Класс", "Train", "Validation", "Test"],
        [[name, counts["train"][i], counts["val"][i], counts["test"][i]] for i, name in enumerate(names)],
    )
    validation = []
    for label, result, provenance in [
        ("E1", metrics["e1-val"], "reports/training/e1/provenance.json"),
        ("E2", metrics["e2-val"], "reports/training/e2/provenance.json"),
        ("E3", e3, "reports/improvement/training/provenance.json"),
    ]:
        validation.append(
            [
                label,
                f"{result['map50']:.4f}",
                f"{result['map50_95']:.4f}",
                f"{read(provenance)['elapsed_seconds'] / 60:.1f}",
            ]
        )
    replacements["VALIDATION_TABLE"] = table(
        ["Опыт", "mAP@0.5", "mAP@0.5:0.95", "Время запуска обучения, мин"], validation
    )
    before = {r["class_id"]: r for r in metrics["e2-val"]["classes"]}
    replacements["E3_CLASSES"] = table(
        ["Класс", "Объектов val", "AP50–95 E2", "AP50–95 E3", "Precision E3", "Recall E3"],
        [
            [
                r["class"],
                r["objects"],
                f"{before[r['class_id']]['ap50_95']:.4f}",
                f"{r['ap50_95']:.4f}",
                f"{r['precision']:.4f}",
                f"{r['recall']:.4f}",
            ]
            for r in e3["classes"]
        ],
    )
    replacements["TEST_TABLE"] = table(
        ["Опыт", "Test mAP@0.5", "Test mAP@0.5:0.95"],
        [
            [
                key.upper(),
                f"{metrics[key + '-test']['map50']:.4f}",
                f"{metrics[key + '-test']['map50_95']:.4f}",
            ]
            for key in ("e1", "e2")
        ],
    )
    sections = []
    for key in ("e1", "e2"):
        sections.append(
            f"**{key.upper()}: показатели по классам на test**\n\n"
            + table(
                ["Класс", "Объектов", "Precision", "Recall", "AP50", "AP50–95"],
                [
                    [
                        r["class"],
                        r["objects"],
                        *[f"{r[k]:.4f}" for k in ("precision", "recall", "ap50", "ap50_95")],
                    ]
                    for r in metrics[key + "-test"]["classes"]
                ],
            )
        )
    replacements["TEST_CLASSES"] = "\n\n".join(sections)
    system = read("reports/selected/system.json")
    rows = [
        ["Полный вызов, p50", f"{system['latency_p50_ms']:.2f} мс"],
        ["Полный вызов, p95", f"{system['latency_p95_ms']:.2f} мс"],
        ["Пропускная способность", f"{system['throughput_fps']:.2f} кадра/с"],
        ["Наблюдённый RSS", f"{system['observed_peak_rss_mb']:.1f} МБ"],
    ]
    for key, label in [
        ("decode_ms", "Декодирование, среднее"),
        ("preprocess", "Подготовка тензора и размера, среднее"),
        ("inference", "Нейросеть, среднее"),
        ("postprocess", "Последующая обработка, среднее"),
    ]:
        rows.append([label, f"{system['stages_mean_ms'][key]:.2f} мс"])
    replacements["SYSTEM_TABLE"] = table(["Показатель", "Значение"], rows)
    robust = read("reports/selected/robustness.json")
    labels = {
        "normal": "Исходные изображения",
        "dark": "Затемнение",
        "bright": "Осветление",
        "occluded": "Искусственное перекрытие",
    }
    replacements["ROBUSTNESS_TABLE"] = table(
        ["Условия", "TP", "FP", "FN", "Precision", "Recall"],
        [
            [labels[k], r["tp"], r["fp"], r["fn"], f"{r['precision']:.3f}", f"{r['recall']:.3f}"]
            for k, r in robust.items()
        ],
    )
    diagnostics = {r["index"]: r for r in read("reports/selected/diagnostics.json")}
    examples = [
        (
            0,
            "critical-fn",
            "Малый объект на сложном фоне",
            "Свод тоннеля распознан как большая область Person, а маленький работник пропущен. Вероятные причины — масштаб, необычный верхний ракурс и недостаток похожих обучающих сцен.",
        ),
        (
            2,
            "critical-fn",
            "Затенённое лицо",
            "Человек и каска найдены, но отрицательные классы СИЗ пропущены. Лицо отвернуто и затенено; проверка отрицательной метки по такому изображению неоднозначна.",
        ),
        (
            3,
            "critical-fn",
            "Съёмка сверху",
            "Далёкие люди занимают мало пикселей; пропущены 13 размеченных объектов. Следует увеличивать разнообразие дальних сцен и ограничивать автоматические выводы по маленьким головам.",
        ),
        (
            4,
            "high-confidence-fp",
            "Повторные рамки конусов",
            "На части конусов выданы дубли, дальние конусы пропущены. Диагностика подтвердила влияние confidence и NMS. Возможную неполноту GT должен проверить человек.",
        ),
        (
            5,
            "class-confusion",
            "Путаница головных уборов",
            "Велосипедный шлем принят за строительную каску. Это ошибка классификации и сдвиг предметной области; нужны трудные отрицательные примеры похожих головных уборов.",
        ),
        (
            6,
            "critical-fn",
            "Предмет у лица",
            "Рука и бутылка возле лица вызывают ложный детект Mask. Перекрытие и похожая форма затрудняют классификацию; такой случай требует человеческого контроля.",
        ),
        (
            7,
            "critical-fn",
            "Велосипедист вне строительного домена",
            "Человек обнаружен, но модель путает велосипедный шлем с Hardhat и пропускает часть отрицательных СИЗ. Неоднородность датасета влияет на смысл результата.",
        ),
        (
            8,
            "critical-fn",
            "Пропуск головы без каски",
            "Маска найдена, NO-Hardhat пропущен, на торсе остаётся лишняя рамка. Понижение порога устраняет FN, но увеличивает FP: одного изменения порога недостаточно.",
        ),
        (
            11,
            "low-iou",
            "Дубли и неточная геометрия",
            "Все GT сопоставлены, но остаются две лишние рамки. Более строгий NMS уменьшает дубли, одновременно создавая пропуск. Требуется баланс на validation.",
        ),
        (
            14,
            "critical-fn",
            "Наушник вместо каски",
            "Наушник распознан как Hardhat, часть объектов на лице пропущена. Похожая форма и профильный ракурс вызывают ошибку модели, которую только NMS не исправляет.",
        ),
    ]
    parts = []
    for number, (index, category, title, explanation) in enumerate(examples, 10):
        r = diagnostics[index]
        parts.append(
            f"### 7.{number - 9}. {title}\n\n![Ошибка E2, кадр {index}](selected/{category}/{index:03d}.jpg)\n\n*Рисунок {number} — E2, test, кадр {index}. Зелёные рамки — эталон, розовые — предсказания. FP={len(r['fp'])}, FN={len(r['fn'])}.*\n\n{explanation}"
        )
    replacements["ERROR_EXAMPLES"] = "\n\n".join(parts)
    text = (REPORT / "lab03-template.md").read_text(encoding="utf-8")
    for key, value in replacements.items():
        text = text.replace("{{" + key + "}}", value)
    if "{{" in text:
        raise ValueError("Unfilled report placeholder")
    (REPORT / "lab03-report.md").write_text(text, encoding="utf-8")
    print("Built reports/lab03-report.md from recorded evidence")


if __name__ == "__main__":
    main()
