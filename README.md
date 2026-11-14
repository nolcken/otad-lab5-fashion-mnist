# ЛР5 ОТАД: машинное обучение в облаке (вариант 5, Fashion-MNIST)

Пайплайн: данные Fashion-MNIST загружаются скриптом в объектное хранилище Yandex Object Storage (S3-совместимое),
CNN обучается на облачной ВМ Yandex Compute Cloud (CPU) с чтением данных из бакета, обученная модель сохраняется
в бакет, REST-сервис на FastAPI при старте берёт модель из бакета и отвечает на `POST /predict`.

```
Fashion-MNIST (GitHub zalandoresearch) --upload_data.py--> бакет data/fashion-mnist/*.gz
бакет --train.py (ВМ, CPU)--> бакет models/cnn_v1.pt, models/metrics_v1.json
бакет --app.py (ВМ, uvicorn :8000)--> POST /predict {"pixels": 28x28} -> {"class_id", "class_name", "probability"}
bench.py: 100 запросов -> среднее и p95 времени ответа
```

## Файлы

| файл | назначение |
|---|---|
| `common.py` | клиент S3 (boto3) с авторизацией IAM-токеном сервисного аккаунта ВМ, чтение формата IDX, модель `FashionCNN` |
| `upload_data.py` | скачивает 4 файла набора и кладёт их в бакет |
| `train.py` | обучение CNN (2 свёрточных слоя), метрики, сохранение модели и метрик в бакет |
| `app.py` | FastAPI: `GET /health`, `GET /metrics`, `POST /predict` |
| `bench.py` | замер латентности на 100 тестовых изображениях (среднее, p95) |
| `run_pipeline.sh` | весь пайплайн на ВМ: зависимости, данные, обучение, сервис (systemd), замер |
| `.env.example` | переменные окружения для запуска вне ВМ (ключи в Git не хранятся) |

## Авторизация в хранилище

Статические ключи не используются. К ВМ привязан сервисный аккаунт с ролью `storage.editor`, скрипты получают
его IAM-токен из сервиса метаданных ВМ (`169.254.169.254`) и передают в заголовке `X-YaCloud-SubjectToken`.
Если нужно запустить скрипты на своём компьютере, задайте `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY` статического
ключа (см. `.env.example`), тогда используется обычная подпись S3.

## Как повторить

1. В Yandex Cloud создать сервисный аккаунт с ролью `storage.editor` и приватный бакет.
2. Создать ВМ Ubuntu 22.04 (2 vCPU, 4 ГБ RAM, 20 ГБ диск) с публичным IP, привязанным сервисным аккаунтом
   и открытым входящим TCP 8000. В поле «Пользовательские данные» (cloud-init) указать:
   ```yaml
   #cloud-config
   packages: [python3-venv, git, curl]
   runcmd:
     - git clone https://github.com/nolcken/otad-lab5-fashion-mnist.git /opt/lab5
     - BUCKET=<имя бакета> bash /opt/lab5/run_pipeline.sh 2>&1 | tee /var/log/lab5.log > /dev/ttyS0
   ```
3. Ход выполнения виден в консоли ВМ на вкладке «Последовательный порт» (строки `[lab5 ...]`)
   или в файле `/var/log/lab5.log`. Через несколько минут сервис доступен:
   ```bash
   curl http://<IP>:8000/health
   curl http://<IP>:8000/metrics
   python bench.py --url http://<IP>:8000      # замер с другого компьютера
   ```
   Интерактивная документация API: `http://<IP>:8000/docs`.
4. После сдачи удалить ВМ, бакет и сервисный аккаунт.

## Пример запроса

```bash
python3 - <<'PY'
import requests
img = [[0] * 28 for _ in range(28)]
print(requests.post("http://<IP>:8000/predict", json={"pixels": img}).json())
PY
```

Ответ имеет вид `{"class_id": 8, "class_name": "Bag", "probability": 0.53}`.
