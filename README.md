# LCT 2026 — Street Falcon Vehicle ReID

Приватный репозиторий Engineering Team для задачи повторной идентификации
транспортных средств Street Falcon. Для каждого изображения из `test_query.csv`
система ранжирует автомобили из `test_gallery.csv`, формирует первую десятку и
может отказаться от ответа, если надёжного совпадения нет.

Государственные регистрационные знаки не используются. Модель работает только
с визуальным представлением размеченного bounding box автомобиля.

## Что реализовано

- безопасная распаковка и проверка исходного архива;
- identity-disjoint разбиение: идентичности в train и validation не пересекаются;
- validation query/gallery с межкамерными позитивами и open-set запросами;
- ResNet-50, ImageNet pretraining, классификационный и batch-hard triplet loss;
- подбор порога отказа по validation F1;
- формирование `submission.csv`, `candidates.csv` и `embeddings.npy`;
- проверка формата результата и упаковка submission;
- локальная веб-панель с метриками, графиками и просмотром top-10;
- GPU Docker-окружение и одна команда для полного прогона.

Первый полный прогон на ревизии `93f4287` завершён. На identity-disjoint
validation он получил `mAP@10 = 0,4523`, `Rank-1 = 0,4837`, `Rank-5 = 0,6463` и
open-set `F1 = 0,9298`. Это локальная validation, а не результат скрытого
leaderboard. Конфигурация, ограничения и checksums зафиксированы в
[`docs/results/baseline-93f4287.md`](docs/results/baseline-93f4287.md).

## Данные

Архив организатора хранится вне Git. На сервере `home` используется каталог:

```text
/home/andrey/datasets/lct26-street-falcon-reid/
├── source/dataset.zip
├── source/evaluate.py
├── source/example_submission.zip
├── extracted/
└── runs/
```

Рекомендуемое значение переменной окружения:

```bash
export LCT_DATA_DIR=/home/andrey/datasets/lct26-street-falcon-reid
```

Архив, изображения, веса, эмбеддинги, результаты и учётные данные запрещено
добавлять в Git, issue, CI-артефакты и Docker image. Подробности находятся в
[`docs/data.md`](docs/data.md).

## Быстрый запуск на A6000

Сборка окружения:

```bash
docker compose build baseline
```

Полный цикл от архива до submission:

```bash
export GIT_COMMIT=$(git rev-parse HEAD)
LCT_DATA_DIR=/home/andrey/datasets/lct26-street-falcon-reid \
docker compose run --rm baseline run \
  --archive /data/source/dataset.zip \
  --data-dir /data/extracted \
  --run-dir /data/runs/resnet50-baseline \
  --config configs/baseline.toml
```

Команда проверяет SHA-256 архива, распаковывает данные, обучает модель,
калибрует open-set порог, выполняет инференс и проверяет выходные файлы.
Повторная распаковка уже проверенного архива пропускается.

Результат:

```text
runs/resnet50-baseline/
├── checkpoint-best.pt
├── history.jsonl
├── split.json
├── validation-report.json
└── submission/
    ├── submission.csv
    ├── candidates.csv
    ├── embeddings.npy
    ├── run-metadata.json
    └── submission.zip
```

## Запуск по этапам

Локальное окружение Python 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

Подготовка:

```bash
street-falcon-reid prepare \
  --archive "$LCT_DATA_DIR/source/dataset.zip" \
  --data-dir "$LCT_DATA_DIR/extracted"
```

Обучение и калибровка:

```bash
street-falcon-reid train \
  --data-dir "$LCT_DATA_DIR/extracted" \
  --run-dir "$LCT_DATA_DIR/runs/resnet50-baseline" \
  --config configs/baseline.toml
```

Инференс и проверка:

```bash
street-falcon-reid infer \
  --data-dir "$LCT_DATA_DIR/extracted" \
  --checkpoint "$LCT_DATA_DIR/runs/resnet50-baseline/checkpoint-best.pt" \
  --output-dir "$LCT_DATA_DIR/runs/resnet50-baseline/submission" \
  --config configs/baseline.toml

street-falcon-reid verify \
  --data-dir "$LCT_DATA_DIR/extracted" \
  --output-dir "$LCT_DATA_DIR/runs/resnet50-baseline/submission"
```

## Веб-панель результатов

Панель читает уже сформированные артефакты и не запускает повторное обучение.
На сервере она публикуется только на loopback-интерфейсе:

```bash
export LCT_DATA_DIR=/home/andrey/datasets/lct26-street-falcon-reid
export LCT_RUN_ID=resnet50-baseline-93f4287
export LCT_WEB_PORT=27810
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)

docker compose up -d dashboard
curl --fail http://127.0.0.1:27810/api/health
```

Для просмотра со своего компьютера откройте SSH-туннель:

```bash
ssh -N -L 8780:127.0.0.1:27810 home_in
```

После этого интерфейс доступен по адресу `http://127.0.0.1:8780`. Датасет
монтируется в контейнер в режиме read-only, а наружу порт не публикуется.

Официальный `evaluate.py` требует скрытый ground truth, поэтому локально им
нельзя получить финальный leaderboard score. Собственная validation использует
тот же контракт `mAP@10`, Rank-1, Rank-5 и режим отказа.

## Smoke-проверка

Чтобы проверить весь код без полноценного обучения:

```bash
street-falcon-reid run \
  --archive "$LCT_DATA_DIR/source/dataset.zip" \
  --data-dir "$LCT_DATA_DIR/extracted" \
  --run-dir "$LCT_DATA_DIR/runs/smoke" \
  --config configs/smoke.toml
```

Smoke-конфигурация использует малую подвыборку, случайную инициализацию и два
шага обучения. Её результаты нельзя сравнивать с рабочим baseline.

## Контракт эксперимента

Каждый результат должен сохранять:

- SHA-256 и версию датасета;
- split revision и seed;
- конфигурацию модели и обучения;
- commit исходного кода;
- validation metrics и подобранный порог;
- checksum итогового checkpoint и submission-файлов.

Рабочий профиль репозитория — `research-python`; Engineering Standards закреплены
на версии `0.2.9`. Веб-панель является внутренним read-only представлением
артефактов baseline и не меняет исследовательский контракт или статус метрик.

## Online search API (maxim_backend)

The API accepts an image and a **client-supplied** bounding box `x, y, w, h`.
It does not detect vehicles or infer bounding boxes. Coordinates are integer pixels
in the original decoded image (no EXIF rotation): x/y are non-negative, w/h are
positive, and the complete rectangle must fit inside the image. After validation,
the checkpoint's crop margin is applied exactly as in the baseline (5% for the
current run), clipped to the image edges.

Install in a separate Python 3.11/3.12 environment:

```bash
python -m venv .venv-api
source .venv-api/bin/activate
python -m pip install -e '.[api,test]'
export LCT_DATA_DIR=/home/andrey/datasets/lct26-street-falcon-reid
export LCT_RUN_DIR="$LCT_DATA_DIR/runs/resnet50-baseline-93f4287"
export LCT_DEVICE=cpu
python -m uvicorn street_falcon_reid.api:create_app --factory --host 127.0.0.1 --port 27812 --workers 1
```

CPU is the conservative default on a shared server; explicitly set
`LCT_DEVICE=cuda` only when GPU resources are available. Use one worker:
each process loads its own model and gallery. This is a separate service from
the dashboard. No existing containers or images need rebuilding.

On the client computer:

```bash
ssh -N -L 8782:127.0.0.1:27812 hackathon-lunopopicks
```

Interactive API documentation: http://127.0.0.1:8782/docs.
Health: `GET /api/v1/health`.

Example request (replace coordinates with the actual client-selected rectangle):

```bash
curl --fail-with-body http://127.0.0.1:8782/api/v1/search \
  -F 'image=@car.jpg' -F x=100 -F y=50 -F w=300 -F h=200 -F top_k=10
```

JPEG and PNG are accepted. Limits: 10 MiB image, 11 MiB complete multipart
request, 20 million decoded pixels; top_k is 1..10 and capped by gallery size.
The API returns `accepted`, `threshold`, `matches` (rank, image_id, score),
`model_version` (checkpoint SHA-256) and `gallery_version` (run ID plus embeddings
SHA-256). Scores are cosine similarities, not probabilities. The threshold is
inherited from baseline validation, not calibrated for arbitrary new domains.
A refused query still returns top-k for inspection.

Errors: 422 invalid/missing bbox or top_k, 413 excessive size, 415 undecodable or
unsupported image, 429 inference already busy, 503 not ready, 500 inference failure.
No upload is retained by application code; temporary multipart upload files are
closed after the request. Model execution is serialized with a non-blocking lock.
This local API has no authentication and must remain on loopback / behind SSH.

Startup reads checkpoint, run config, run-metadata.json, verification.json,
embeddings.npy and test_gallery.csv. It checks model identity, preprocessing,
threshold, dimensions, embedding checksum and gallery validity. Only the gallery
slice after query_count is retained; gallery images are not re-embedded.
New offline runs record gallery_csv_sha256. Legacy runs lack this checksum:
startup emits a warning and requires the original, unmodified gallery CSV in
its original order. Its order cannot be independently proven from legacy metadata.
Startup fails if required artifacts are missing or incompatible.

Checks in an isolated environment:

```bash
python -m ruff check .
python -m pytest -q
python scripts/smoke.py
python scripts/smoke_online.py --data-dir "$LCT_DATA_DIR/extracted" --run-dir "$LCT_RUN_DIR"
```

The last command compares online and batch embeddings on the same CPU at 1e-6
tolerance, then compares top-k with the saved GPU run (score tolerance 1e-3
allows differences between devices/batch sizes). It checks one real query
against saved rankings/scores on CPU. It writes no dataset/run files and does
not measure full-dataset quality or production performance.
