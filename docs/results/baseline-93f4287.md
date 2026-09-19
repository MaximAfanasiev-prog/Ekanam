# Baseline `93f4287`

## Статус

Полный 30-эпоховый прогон завершён 20 сентября 2026 года на NVIDIA RTX A6000.
Результат относится только к локальной identity-disjoint validation. Скрытый
ground truth организатора недоступен, поэтому leaderboard score не заявляется.

## Воспроизводимость

| Поле | Значение |
|---|---|
| Git commit | `93f4287e6ed05ac310b75db58de39c205b6605f5` |
| Конфигурация | `configs/baseline.toml` |
| Dataset SHA-256 | `a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613` |
| Split seed | `20260919` |
| Train | 7 651 объектов |
| Validation query | 308 объектов: 246 closed-set и 62 open-set |
| Validation gallery | 1 277 объектов |
| Docker image | `sha256:ccefc0cff92ef78c47b720d9b6f1f41e2ebbe7569c4883efb1c75808154754b8` |
| PyTorch | `2.5.1+cu124` |

Train и validation identities не пересекаются. Для closed-set validation
позитивы находятся на другой камере; 62 open-set запроса не имеют позитивов в
gallery.

## Выбранный checkpoint

Лучшим по `mAP@10` стал checkpoint 10-й эпохи.

| Метрика | Значение |
|---|---:|
| mAP@10 | 0,4523 |
| Rank-1 | 0,4837 |
| Rank-5 | 0,6463 |
| Open-set Precision | 0,9259 |
| Open-set Recall | 0,9336 |
| Open-set F1 | 0,9298 |
| Open-set TNR | 0,7903 |
| Порог cosine similarity | 0,4955 |

Checkpoint занимает 101 107 760 байт, embedding dimension — 512. SHA-256
checkpoint:

```text
64879053616e87e0ea8d7af7ae6e5417048a91e5d5ad54ee70236e718deeb3f5
```

## Submission

Инференс обработал 1 110 query и 750 gallery. Форматная проверка прошла;
`submission.zip` содержит ровно три файла и проходит `unzip -t`.

| Файл | SHA-256 |
|---|---|
| `submission.csv` | `e44cf89bafa3d06a0110c11f1dea318621a06c3eb4ba2052788f9a084daf47fb` |
| `candidates.csv` | `69ed45bfbf05c77a57911aa56aa827710ca39f9a9e9932044ca7c6a48bdb1924` |
| `embeddings.npy` | `7a76cbf4769335b0d085b998df43ec5542779cde9bb41110bbe624784f580c63` |

Артефакты находятся вне Git:

```text
/home/andrey/datasets/lct26-street-falcon-reid/runs/resnet50-baseline-93f4287/
```

Датасет, checkpoint, embeddings и submission не публикуются без отдельного
решения по условиям организатора и доступу.
