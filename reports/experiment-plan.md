# План и журнал

## Исходное состояние

- Upstream: https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection
- Исходный коммит: 19b7322 (точный SHA фиксируется в provenance).
- Форк: https://github.com/LizynBr/Construction-Site-Safety-PPE-Detection
- В исходном репозитории есть веса и результаты, но нет исполняемого ноутбука/скрипта обучения и самих изображений train/valid/test.
- data/ppe_data.yaml содержит абсолютные пути Kaggle; data/data.yaml — пути вне каталога data.
- Исходные результаты не являются результатами лабораторной. best.pt уже обучен на исходном датасете и не используется для честного сравнения E1/E2.
- Данные: Construction Site Safety v28, Roboflow Universe Projects; зеркало snehilsanyal на Kaggle; заявлена CC BY 4.0.
- Исходный датасет содержит готовые аугментации (5 версий), поэтому случайное разбиение файлов недопустимо.

## Ресурсы

Windows, Intel Core Ultra 5 125H, 16 GB RAM, Intel Arc Graphics. Обучение локально на CPU.
Параметры двух опытов будут зафиксированы после проверки скорости, до сравнительного обучения и тестовой оценки.
Подбор и выбор модели — только по validation. Итоговая test-оценка — после фиксации выбора.
