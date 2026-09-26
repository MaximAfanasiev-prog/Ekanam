# Что передать команде бэкенда

## Модель

Модель получает RGB crop автомобиля и возвращает L2-нормализованный
float32 embedding фиксированной размерности D (текущий baseline: D=512).
Детектор не нужен: bbox всегда приходит от клиента как x, y, w, h.

Текущий загрузчик поддерживает ResNet-18/ResNet-50 и checkpoint:

- model: state_dict;
- model_config: backbone, embedding_dim, num_classes;
- data_config: image_height, image_width, crop_margin;
- open_set_threshold: конечное число, подобранное на validation.

Если меняется архитектура, структура checkpoint или нормализация изображения,
нужен адаптер и проверка совместимости. Нельзя просто переименовать OSNet checkpoint
в checkpoint-best.pt. Поддержка произвольной модели не заявляется.
Checkpoint загружается с weights_only=True; пользователь API не может загрузить свои веса.

## Артефакты одного прогона

Передать вне Git:

```text
run/
  checkpoint-best.pt
  config.toml
  submission/
    embeddings.npy
    run-metadata.json
    verification.json
data/
  test_gallery.csv
  test_query.csv
  images/
```

Для экспорта галереи test_query.csv и images не нужны; они нужны для
контрольного сравнения полного прогона и HTTP-smoke.

embeddings.npy: float32 [query_count + gallery_count, D], сначала query,
затем gallery. Строки галереи строго в порядке test_gallery.csv.
Все значения конечны, длина каждого вектора равна 1 с допуском 1e-3.
ID в gallery CSV должны быть уникальными, непустыми.
CSV: image_id,x,y,w,h. Номера автомобилей не используются как признаки.

run-metadata.json: checkpoint_sha256, embedding_dim, query_count,
gallery_count, open_set_threshold, gallery_csv_sha256.
Новая функция run_inference уже записывает gallery_csv_sha256.
verification.json: embedding_shape и sha256 для embeddings.npy.
config.toml должен содержать параметры фактической обработки изображений.

После изменения весов, crop/resize/нормализации пересчитать всю галерею.
Порог нового baseline проверяет ML-команда; backend только применяет его.
Совпадение порога с прежней моделью не предполагается.

## Пакет галереи для сервиса

Команда конвертирует существующий прогон без обучения и без пересчёта embeddings:

```bash
python -m street_falcon_reid.gallery_bundle \
  --data-dir /path/to/data \
  --run-dir /path/to/run \
  --output /private/artifacts/gallery-v2
```

Выход:

```text
gallery-v2/
  manifest.json
  ids.json
  gallery.npy
```

ids.json — упорядоченный список ID; gallery.npy — только галерея [N,D].
manifest.json связывает контрольными суммами IDs и embeddings с конкретным
checkpoint, размерностью, порогом и параметрами обработки. Сервис проверяет
всё при старте и не нуждается в исходных CSV или изображениях.
Порядок IDs и embeddings нельзя менять по отдельности.

Выходной каталог не должен существовать: для каждой версии используется
новое имя. manifest.json записывается последним; неполный пакет не загрузится.
Исходные данные не меняются. Пакеты, веса и изображения не добавляются в Git,
PR, публичные логи или CI-артефакты.

Старый прогон без gallery_csv_sha256 допускается только с флагом --allow-legacy
и исходным CSV в исходном порядке. В manifest фиксируется
source_csv_checksum_verified=false. Это фиксирует выбранную пару файлов
на будущее, но не доказывает правильность первоначального порядка.
Для нового baseline флаг не нужен и использовать его не следует.

## Минимальная приёмка от ML-команды

- Указаны версия датасета, split revision, конфигурация и validation-метрики.
- Checkpoint загружается; размерность и обработка совпадают с галереей.
- Есть известный контрольный запрос с bbox и ожидаемым top-k.
- Порог отказа рассчитан на identity-disjoint validation.
- Пакет проходит проверку контрольных сумм и HTTP-smoke.
- Отдельно обсуждены лицензия/доступ к данным и ограничения интерпретации метрик.

Метрики текущего baseline не являются оценкой новой модели или скрытого leaderboard.

## Экспорт через готовый API-образ

Если на сервере нет Python-окружения, использовать собранный Dockerfile.api.
В примере LCT_DATA_DIR — корень датасета с extracted/ и runs/; NEW_RUN заменить
на имя нового прогона. Для актуального baseline не применять --allow-legacy.

```bash
export ARTIFACTS="$HOME/artifacts/street-falcon"
mkdir -p "$ARTIFACTS"
docker run --rm --network none --read-only \
  --cpus 2 --memory 3g --tmpfs /tmp:size=64m,mode=1777 \
  --user "$(id -u):$(id -g)" \
  -v "$LCT_DATA_DIR:/data:ro" -v "$ARTIFACTS:/artifacts" \
  --entrypoint python lct26-street-falcon-reid:api-maxim \
  -m street_falcon_reid.gallery_bundle \
  --data-dir /data/extracted --run-dir /data/runs/NEW_RUN \
  --output /artifacts/gallery-v2
```
