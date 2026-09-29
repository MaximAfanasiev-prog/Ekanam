# YOLO-cls embedding (без дообучения)

Независимая ветка эксперимента: предобученные классификаторы Ultralytics YOLOv8-cls и YOLO26-cls
используются как замороженные экстракторы признаков ТС. Модуль не импортирует ничего из ResNet-ветки
команды и не зависит от её конфигов.

Результаты и рекомендация: [`docs/results/yolo-comparison.md`](../../docs/results/yolo-comparison.md).
Рекомендованная модель — **`yolo26l-cls`**, препроцессинг `center_crop`, порог candidates **0.8485**.

## Быстрый старт

```bash
pip install -r models/yolo_embedding/requirements.txt
python -m models.yolo_embedding.download_weights                  # скачать и проверить sha256 (нужна сеть)

# сабмит по официальному тесту (офлайн, веса локальные)
python -m models.yolo_embedding.predict --model yolo26l-cls --out submissions/yolo_embedding/yolo26l-cls

# воспроизведение сравнения
python -m models.yolo_embedding.make_split                 # data/mini_*.csv, data/mini_val_split.json (seed 42)
python -m models.yolo_embedding.run_minival --official     # 10 моделей × 3 препроцессинга -> docs/results/yolo_minival.json
python -m models.yolo_embedding.benchmark                  # скорость -> docs/results/yolo_speed.json
```

Все команды запускаются из корня репозитория. Путь к датасету задаётся переменной `FALCON_DATASET_DIR`
(по умолчанию `/home/andrey/datasets/lct26-street-falcon-reid/extracted`), путь к официальному скрипту
метрик — `FALCON_EVALUATE_PY` (по умолчанию `../source/evaluate.py` относительно датасета).

## Пайплайн

| Шаг | Файл | Что делается |
|---|---|---|
| Данные | `data.py` | Чтение CSV, кроп по `x, y, w, h` **как есть**, RGB, препроцессинг до 224×224, значения в [0, 1] (Ultralytics-cls не использует mean/std). |
| Эмбеддинг | `extractor.py` | `Classify`-голова = `Conv1x1(c→1280) → GAP → Linear(1280→1000)`. Эмбеддинг — вход последнего `Linear` (выход GAP, 1280-d), снимается forward-hook'ом. Затем L2-нормализация. |
| Поиск | `retrieval.py` | FAISS `IndexFlatIP` по эмбеддингам gallery, top-10 по косинусу. Запись артефактов. |
| Метрики | `metrics.py` | Копия протокола официального `evaluate.py` (junk-фильтр по камере, AP@10 / min(n_pos, 10), open-set исключения, решение по top-1). Сверка с официальным скриптом — флаг `--official`. |
| Сплит | `make_split.py` | Self-made open-set mini-val из `train.csv`. |
| Сравнение | `run_minival.py`, `benchmark.py` | Одинаковый прогон для всех моделей. |
| Сабмит | `predict.py` | Тест → `submission.csv`, `embeddings.npy`, `candidates.csv`, `run_meta.json`. Обработка чанками по 512, память не растёт с размером выборки. |

**Margin вокруг bbox — моё собственное решение, а не требование данных.** Параметр `--margin` есть
в `run_minival.py` и `predict.py`: это относительный отступ с каждой стороны, обрезанный по границам кадра.
**По умолчанию он равен 0**, и все результаты и сабмиты получены с margin = 0. Отдельно он не исследовался.

**Препроцессинг** (`--mode`): `center_crop` (родной для Ultralytics-cls, по умолчанию), `squash`, `letterbox`.
Выбран `center_crop`: он лучший в среднем по 10 моделям и 3 сплитам.

## Формат артефактов

Сверено с `dataset/README.md` и официальным `evaluate.py`:

- `submission.csv` — `query_id,gallery_id_1,...,gallery_id_10`, по строке на каждый query из
  `test_query.csv`, **без заголовка**. Так в официальном скрипте («без заголовка») и в примере организаторов.
- `embeddings.npy` — float32 `[N_query + N_gallery, 1280]`: сначала все `test_query.csv`, затем все
  `test_gallery.csv`, в порядке строк файлов. Уже L2-нормированы.
- `candidates.csv` — с заголовком `query_id,gallery_id,confidence`. `confidence` — косинусное сходство,
  обрезанное в [0, 1]. Строка пишется для каждого кандидата из top-10 со сходством ≥ порога.
  **Отказ = для этого `query_id` нет ни одной строки.** Это прямо подтверждено официальным скриптом
  («Отказ кодируется ОТСУТСТВИЕМ строк для этого query_id»).

## Порог для candidates.csv

Порог выбирается на self-made mini-val (seed 42) как максимизирующий F1 решения «есть совпадение / отказ»
по протоколу организаторов: TP — ответ дан и top-1 верный, FP — ответ дан, но top-1 неверный или пары нет,
FN — отказ при наличии пары, TN — отказ без пары. Порог ставится посередине между оптимальным top-1 скором
и ближайшим меньшим. Для `yolo26l-cls` это **0.8485**: F1 = 0.778, Precision = 0.821, Recall = 0.740,
TNR = 0.704.

Максимум F1 плоский в диапазоне 0.84–0.86, а TNR там растёт от 0.58 до 0.78. Поэтому стоит заранее знать
компромисс: порог 0.87 даёт TNR 0.87 при F1 0.767 (на 0.011 ниже максимума). Меняется флагом `--threshold`.
Полная кривая — в `docs/results/yolo-comparison.md` и `docs/results/yolo_minival.json`
(`threshold_curve`). Порог подобран на mini-val, а доля запросов без пары в тесте неизвестна.
На тесте этот порог принимает 85% запросов, на mini-val — 66%.

## Внешние ресурсы, версии, лицензии

**Веса** — официальные предобученные классификаторы Ultralytics (ImageNet-1k, 224×224, 200 эпох),
релиз `v8.4.0` репозитория [`ultralytics/assets`](https://github.com/ultralytics/assets/releases/tag/v8.4.0),
URL вида `https://github.com/ultralytics/assets/releases/download/v8.4.0/<model>.pt`.
SHA256 всех файлов — в `weights/SHA256SUMS`.

| Файл | Размер | Обучен (метаданные чекпойнта) | SHA256 |
|---|---|---|---|
| `yolo26l-cls.pt` (рекомендован) | 27.2 МБ | ultralytics 8.3.100, 2025-10-24 | `7a3aebb7…986ee129e` |
| `yolov8l-cls.pt` (альтернатива) | 71.7 МБ | ultralytics 8.0.174, 2023-09-12 | `d8c56328…3dda20bd84` |

Остальные 8 весов (`yolov8{n,s,m,x}-cls`, `yolo26{n,s,m,x}-cls`) нужны только для воспроизведения сравнения.
В git они не хранятся и скачиваются `download_weights.py`. Все 10 вместе весят 366 МБ, что меньше лимита 2 ГБ.

**Лицензия: AGPL-3.0.** И код Ultralytics (`ultralytics` 8.4.163), и веса YOLOv8/YOLO26 распространяются
под AGPL-3.0 (https://ultralytics.com/license; строка `license` в метаданных чекпойнтов). Это внешний ресурс
с copyleft-лицензией: при распространении решения на его основе действуют требования AGPL-3.0.
Коммерческое использование без раскрытия кода требует Ultralytics Enterprise License.

Библиотеки, на которых получены результаты: Python 3.12.3, torch 2.6.0+cu124, torchvision 0.21.0+cu124,
ultralytics 8.4.163, faiss-cpu 1.15.1, numpy 2.5.2, pandas 3.0.6, pillow 12.3.0 (`requirements.txt`).
Железо: NVIDIA RTX A6000 48 ГБ, 32 CPU.

Инференс не ходит в сеть: веса локальные, при импорте ставится `YOLO_OFFLINE=1`, и методы Ultralytics,
которые шлют телеметрию (`predict`/`train`), не вызываются — используется только `nn.Module`.

## Чего модуль не делает

- Детекция, распознавание номеров, трекинг — bbox даны организаторами. Номерные знаки не используются никак.
- Обучение и дообучение YOLO — вне рамок этой ветки.
- Сравнение с EfficientNet или ResNet без triplet — делают другие члены команды.
- Dockerfile, docker-compose и API — общая интеграция команды, этот модуль их не трогает.
