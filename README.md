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
на версии `0.2.9`. API и браузерный интерфейс могут появиться отдельным проектным
этапом, но не входят в контракт этого baseline и не заявлены как готовые.
