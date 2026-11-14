#!/bin/bash
# Полный пайплайн на облачной ВМ: данные -> бакет -> обучение -> модель в бакет -> REST-сервис -> замер.
# Запускается из cloud-init при создании ВМ: BUCKET=<имя> bash run_pipeline.sh
set -e
cd "$(dirname "$0")"
export BUCKET S3_ENDPOINT=${S3_ENDPOINT:-https://storage.yandexcloud.net} S3_REGION=${S3_REGION:-ru-central1}
log() { echo "[lab5 $(date +%H:%M:%S)] $*"; }

log "установка зависимостей"
python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q -r requirements.txt
./venv/bin/python -c "import torch, platform; print('python', platform.python_version(), 'torch', torch.__version__, 'cpu', torch.get_num_threads())"

log "шаг 1: загрузка данных в бакет $BUCKET"
./venv/bin/python upload_data.py

log "шаг 2: обучение CNN"
./venv/bin/python train.py --epochs 5

log "шаг 3: запуск сервиса"
cat > /etc/systemd/system/lab5.service <<UNIT
[Unit]
Description=Fashion-MNIST CNN REST service
After=network-online.target

[Service]
WorkingDirectory=$(pwd)
Environment=BUCKET=$BUCKET S3_ENDPOINT=$S3_ENDPOINT S3_REGION=$S3_REGION
ExecStart=$(pwd)/venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now lab5.service
for i in $(seq 1 30); do curl -sf localhost:8000/health && break; sleep 2; done
echo

log "шаг 4: 100 запросов к сервису"
./venv/bin/python bench.py --url http://localhost:8000
log "готово"
