# Runbook baseline

## Текущее состояние

В репозитории реализован воспроизводимый модельный baseline: подготовка архива,
identity-disjoint validation, обучение, калибровка отказа, инференс, проверка и
упаковка submission. Постоянного deployment, API и браузерного интерфейса пока
нет. Доступность каких-либо сервисов не заявляется.

## Требования

- NVIDIA GPU с CUDA и не менее 16 ГБ VRAM;
- Docker с NVIDIA Container Runtime и Docker Compose;
- закрытый архив организатора вне checkout;
- не менее 12 ГБ свободного места для архива, распакованных данных и результатов.

Проверка GPU:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

## Полный запуск

```bash
export LCT_DATA_DIR=/home/andrey/datasets/lct26-street-falcon-reid
export GIT_COMMIT=$(git rev-parse HEAD)
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)

docker compose build baseline
docker compose run --rm baseline run \
  --archive /data/source/dataset.zip \
  --data-dir /data/extracted \
  --run-dir /data/runs/resnet50-baseline \
  --config configs/baseline.toml
```

Исходный архив проверяется по SHA-256 до распаковки. При совпадении
`.prepared.json` повторная распаковка пропускается.

## Критерии приёмки прогона

- `validation-report.json` содержит checksum checkpoint и устройство `cuda`;
- `history.jsonl` содержит запись для каждой эпохи;
- train и validation identities не пересекаются согласно `split.json`;
- `verification.json` подтверждает 1 110 query, 750 gallery и форму embeddings;
- `submission.zip` содержит только `submission.csv`, `candidates.csv` и
  `embeddings.npy`;
- размер checkpoint меньше ограничения организатора 2 ГБ;
- в Git diff нет данных, весов, эмбеддингов и credentials.

Финальный score нельзя получить локально: официальный ground truth скрыт.
`evaluate.py` применяется организаторами после подачи.

## Повторный инференс

```bash
docker compose run --rm baseline infer \
  --data-dir /data/extracted \
  --checkpoint /data/runs/resnet50-baseline/checkpoint-best.pt \
  --output-dir /data/runs/resnet50-baseline/submission \
  --config configs/baseline.toml
```

## Сбой и восстановление

- Ошибка checksum: не распаковывать архив; повторно получить исходный файл и
  сверить provenance.
- Остался `.extracted.extracting`: проверить причину прерывания. Код намеренно не
  удаляет этот каталог автоматически.
- CUDA OOM: уменьшить `identities_per_batch`, сохранив не менее двух экземпляров
  одной идентичности, и записать изменение как новую конфигурацию эксперимента.
- Повреждён checkpoint: использовать предыдущий файл с подтверждённым SHA-256;
  не перезаписывать evidence успешного прогона.
- Для полного отката удалить только новый именованный каталог прогона после
  проверки точного пути. Исходный архив и `extracted/` не затрагивать.

## Граница доставки

Прямой deployment не разрешён. Development и production environments в
`catalog-info.yaml` не зарегистрированы. API, веб-интерфейс и GitOps-доставка
добавляются отдельным PR после определения namespace, data boundary, health
endpoint и процедуры приёмки.
