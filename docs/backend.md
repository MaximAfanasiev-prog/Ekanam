# Бэкенд хакатона: запуск и передача фронтенду

Рабочая ветка: maxim_backend. Это внутренний API поиска автомобиля по
клиентскому bbox. Публикация наружу и объединение с main — отдельные решения.
Существующая панель результатов не заменяется этим API.

## Контракт фронтенда

POST /api/v1/search, multipart/form-data:

- image: один статический JPEG или PNG, до 10 MiB и 20 млн пикселей;
- x, y: целые координаты левого верхнего угла, от нуля;
- w, h: целые положительные ширина и высота;
- top_k: целое от 1 до 10, по умолчанию 10.

Прямоугольник должен целиком находиться внутри исходного изображения.
Координаты относятся к пикселям файла без автоматического EXIF-поворота.
Если интерфейс показывает уменьшенную картинку, фронтенд пересчитывает
выбранный прямоугольник в исходные пиксели. Backend не обнаруживает автомобиль
и не вычисляет bbox. После проверки применяется crop_margin checkpoint,
как при обучении (текущий baseline: 0.05), с обрезкой отступа по краям кадра.

Ответ:

```json
{
  "request_id": "generated-id",
  "accepted": true,
  "threshold": 0.4955,
  "matches": [
    {"rank": 1, "image_id": "example-gallery-id", "score": 0.82}
  ],
  "model_version": "checkpoint-sha256",
  "gallery_version": "manifest-sha256"
}
```

Числа в примере условные. score — косинусное сходство, не вероятность.
accepted означает, что хотя бы один score в выбранном top_k не ниже порога checkpoint.
Если все кандидаты ниже порога, API возвращает HTTP 200, accepted=false и matches=[].
Это нормальный результат «0 совпадений», а не ошибка. Если хотя бы один проходит,
возвращается весь выбранный топ. decision_score — максимальный cosine в этом топе
до проверки порога, в том числе при отказе.
Ссылки на изображения не выдаются: image_id должен сопоставляться с доступной
фронтенду галереей. API не выдаёт закрытые исходные изображения.

Минимальный вызов с общего домена:

```javascript
const form = new FormData();
form.append("image", file);
for (const [key, value] of Object.entries({x, y, w, h, top_k: 10})) {
  form.append(key, String(value));
}
const response = await fetch("/api/v1/search", {method: "POST", body: form});
const result = await response.json();
// Content-Type вручную не задавать: браузер добавляет multipart boundary.
if (!response.ok) {
  console.error(result.error?.code, result.detail, result.request_id);
}
```

Ошибка содержит detail (совместимость), error.code, error.message и request_id.
При ошибке полей есть error.fields, но нет содержимого загруженного файла.
X-Request-ID возвращается также в заголовке, X-Process-Time-Ms — время до начала ответа.

| HTTP | error.code | Действие фронтенда |
|---|---|---|
| 400 | bad_request | Исправить структуру запроса |
| 408 | upload_timeout | Повторить загрузку при стабильном соединении |
| 413 | payload_too_large | Уменьшить файл или разрешение |
| 415 | unsupported_image | Выбрать корректный статический JPEG/PNG |
| 422 | validation_error | Исправить bbox или top_k |
| 429 | busy | Показать занятость; повторить не раньше Retry-After |
| 500 | internal_error | Передать оператору request_id |
| 503 | not_ready | Подождать готовности сервиса |

Один процесс принимает не больше двух загрузок и выполняет не больше одного
инференса одновременно. Очереди задач нет. Для 429 допустимы 2–3 повтора
с паузой и небольшим случайным разбросом, а не бесконечные повторы.
Загрузка ограничена 15 секундами, общий multipart — 11 MiB.
Код не сохраняет изображения и embeddings запросов.

## Проверки состояния

- GET /api/v1/live — процесс отвечает;
- GET /api/v1/ready — модель и галерея загружены, стартовый прогрев завершён;
- GET /api/v1/health — совместимый краткий ответ готовности;
- GET /api/v1/info — версии, устройство, размеры и лимиты;
- /docs и /openapi.json — интерактивное описание и контракт.

При невалидной модели или галерее запуск завершается с ошибкой.
Готовность после старта не является непрерывной проверкой GPU вычислений.
В логах — request_id, шаблон маршрута, статус и время; изображений, bbox,
имён загрузок и результатов поиска в журнале запросов нет.

## Отдельный Docker-запуск

Пакет галереи создаётся по инструкции [baseline-contract.md](baseline-contract.md).
Указать реальные абсолютные пути вне репозитория:

```bash
export LCT_CHECKPOINT=/path/to/run/checkpoint-best.pt
export LCT_GALLERY_DIR=/path/to/gallery-bundle-v1
export LOCAL_UID=$(id -u)
export LOCAL_GID=$(id -g)
export LCT_API_PORT=27812

docker compose -p maxim-reid-api -f compose.api.yml build api
docker compose -p maxim-reid-api -f compose.api.yml up -d api
docker compose -p maxim-reid-api -f compose.api.yml ps
curl --fail http://127.0.0.1:27812/api/v1/ready
```

Контейнер работает на CPU, с лимитом 2 CPU и 3 GiB RAM, не от root,
с read-only файловой системой и двумя read-only mounts: checkpoint и gallery.
Исходный датасет не монтируется. Зависимости устанавливаются при сборке образа,
а не при старте. Используется один worker. Логи ротируются.
restart: unless-stopped возобновляет работу после перезапуска Docker/сервера,
если сам Docker запускается автоматически. Статус unhealthy сам по себе
не перезапускает контейнер — причину нужно смотреть в логах.

Опциональный GPU-запуск (после проверки доступных ресурсов):

```bash
docker compose -p maxim-reid-api -f compose.api.yml -f compose.api.gpu.yml up -d api
```

GPU-вариант требует NVIDIA Container Runtime и отдельной проверки с моделью коллег.
Сервис не скачивает веса и не обучается при запуске.

Для доступа с компьютера:

```bash
ssh -N -L 8782:127.0.0.1:27812 hackathon-lunopopicks
```

После этого: http://127.0.0.1:8782/docs.

## Общий домен / прокси

Фронтенд вызывает относительный /api/v1/search. CORS не требуется.
API остаётся на loopback; публичный доступ и права пользователей обеспечивает
внешний прокси. CORS не заменяет авторизацию.

Пример фрагмента Nginx на том же хосте (не применён к серверу автоматически):

```nginx
location /api/v1/ {
    client_max_body_size 11m;
    client_body_timeout 15s;
    proxy_connect_timeout 3s;
    proxy_read_timeout 30s;
    proxy_send_timeout 30s;
    proxy_request_buffering off;
    proxy_pass http://127.0.0.1:27812;
}
```

Если прокси находится в контейнере, его 127.0.0.1 — другой контейнер:
нужен отдельный маршрут к API через Docker-сеть или адрес хоста.
Ошибки, сформированные самим прокси (например 413/502), могут быть не JSON;
фронтенду нужен резервный текст ошибки.
Для размещения за дополнительным префиксом использовать LCT_ROOT_PATH
и согласованное удаление префикса прокси. Для обычного /api/v1 оставить пустым.

## Замена baseline и откат

1. Получить checkpoint и артефакты одного прогона, создать новую версию bundle.
2. Проверить их в отдельном контейнере на порту 27813.
3. Выполнить smoke_http.py с новым query/gallery набором.
4. После успешной проверки переключить LCT_CHECKPOINT/LCT_GALLERY_DIR и пересоздать API.
5. При проблеме вернуть предыдущие пути и прежний образ, затем повторить проверку ready.

Не менять файлы уже работающего bundle. Данные загружаются при старте;
горячая замена не поддерживается. Перед обновлением сохранить ID старого образа:

```bash
docker image inspect lct26-street-falcon-reid:api-maxim --format '{{.Id}}'
docker compose -p maxim-reid-api -f compose.api.yml logs --tail 100 api
docker compose -p maxim-reid-api -f compose.api.yml stop api
```

stop останавливает только этот API. Для воспроизводимого отката пометить
старый образ отдельным тегом и выбрать его через Compose override image.

## Приёмочные проверки

```bash
python -m pip install -c requirements-api.lock -e '.[api,test]'
python -m ruff check .
python -m pytest -q
python scripts/smoke.py
python scripts/smoke_online.py --data-dir "$LCT_DATA_DIR/extracted" --run-dir "$LCT_RUN_DIR"
python scripts/smoke_http.py --url http://127.0.0.1:27812 \
  --data-dir "$LCT_DATA_DIR/extracted" --run-dir "$LCT_RUN_DIR"
```

HTTP-smoke проверяет настоящую загрузку, bbox, top-10, request_id и ограниченную
серию одновременных запросов. Это малая smoke-выборка, не нагрузочная
сертификация и не оценка качества на полном датасете.

До baseline коллег готовы API, ограничения, диагностика, упаковка галереи,
контейнер и контракт. После получения остаются проверка архитектуры, сборка
галереи, проверка качества/порога на их validation и совместный тест фронтенда.
