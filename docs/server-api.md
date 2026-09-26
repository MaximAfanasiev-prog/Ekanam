# Текущий внутренний API на сервере

SSH: hackathon-lunopopicks. Репозиторий и ветка: lct26-street-falcon-reid,
maxim_backend. Compose project: maxim-reid-api. Контейнер: maxim-reid-api-api-1.

Адрес на сервере: http://127.0.0.1:27812.
Для своего браузера: ssh -N -L 8782:127.0.0.1:27812 hackathon-lunopopicks,
затем http://127.0.0.1:8782/docs.

Конфигурация запуска сохранена вне Git:
`/home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/api.env`.

В терминале сервера:

```bash
cd /home/projects/hackathon_2026_lunopopicks/hackathon_maxim/lct26-street-falcon-reid
export API_ENV=/home/projects/hackathon_2026_lunopopicks/hackathon_maxim/artifacts/street-falcon/api.env
docker compose --env-file "$API_ENV" -p maxim-reid-api -f compose.api.yml ps
docker compose --env-file "$API_ENV" -p maxim-reid-api -f compose.api.yml logs --tail 100 api
curl --fail http://127.0.0.1:27812/api/v1/ready
```

Перезапустить только API (короткая пауза в доступности):

```bash
docker compose --env-file "$API_ENV" -p maxim-reid-api -f compose.api.yml restart api
```

Обновлять образ и менять пути модели следует по [backend.md](backend.md)
с предварительной проверкой на отдельном порту. Runtime-конфигурация, галерея,
модель и изображения в Git не добавляются. Рабочая панель на 27810 независима.

Текущая модель: resnet50-baseline-93f4287.
Пакет галереи: artifacts/street-falcon/gallery-baseline-v1 вне checkout.
Это старый baseline с явно зафиксированной legacy-оговоркой о порядке CSV.
Автоперезапуск контейнера настроен; автозапуск Docker после перезагрузки ОС
зависит от настройки сервера. Полная перезагрузка общего сервера не выполнялась.
