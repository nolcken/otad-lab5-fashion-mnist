"""Общие функции: подключение к объектному хранилищу, чтение Fashion-MNIST, модель CNN."""
import gzip
import io
import os

import json
import time
import urllib.request

import boto3
import numpy as np
from botocore import UNSIGNED
from botocore.config import Config
import torch
from torch import nn

CLASSES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

DATA_PREFIX = "data/fashion-mnist/"
MODEL_KEY = "models/cnn_v1.pt"
METRICS_KEY = "models/metrics_v1.json"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def env(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        raise SystemExit(f"Не задана переменная окружения {name} (см. .env.example)")
    return value


METADATA_TOKEN_URL = "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token"
_token = {"value": None, "expires": 0.0}


def iam_token():
    """IAM-токен сервисного аккаунта, привязанного к ВМ (берётся из сервиса метаданных, ключи не нужны)."""
    if _token["value"] is None or time.time() > _token["expires"] - 300:
        req = urllib.request.Request(METADATA_TOKEN_URL, headers={"Metadata-Flavor": "Google"})
        data = json.loads(urllib.request.urlopen(req, timeout=5).read())
        _token["value"] = data["access_token"]
        _token["expires"] = time.time() + int(data.get("expires_in", 3600))
    return _token["value"]


def s3_client():
    """Клиент S3 для Yandex Object Storage.

    Если заданы AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, используется статический ключ.
    Иначе запросы подписываются IAM-токеном сервисного аккаунта ВМ (заголовок X-YaCloud-SubjectToken).
    """
    endpoint = env("S3_ENDPOINT", "https://storage.yandexcloud.net")
    region = env("S3_REGION", "ru-central1")
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        return boto3.client("s3", endpoint_url=endpoint, region_name=region)
    client = boto3.client("s3", endpoint_url=endpoint, region_name=region,
                          config=Config(signature_version=UNSIGNED))

    def add_token(request, **kwargs):
        request.headers["X-YaCloud-SubjectToken"] = iam_token()

    client.meta.events.register("before-send.s3.*", add_token)
    return client


def bucket():
    return env("BUCKET")


def read_idx(raw_gz):
    """Разбор формата IDX (как в исходном архиве Fashion-MNIST)."""
    data = gzip.decompress(raw_gz)
    magic = int.from_bytes(data[0:4], "big")
    if magic == 2051:   # изображения
        n, rows, cols = (int.from_bytes(data[i:i + 4], "big") for i in (4, 8, 12))
        return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(n, rows, cols)
    if magic == 2049:   # метки
        return np.frombuffer(data, dtype=np.uint8, offset=8)
    raise ValueError(f"неизвестный формат IDX: {magic}")


def load_from_bucket(s3, name):
    obj = s3.get_object(Bucket=bucket(), Key=DATA_PREFIX + FILES[name])
    return read_idx(obj["Body"].read())


class FashionCNN(nn.Module):
    """Простая CNN: два свёрточных блока и полносвязный классификатор."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32 x 14 x 14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 64 x 7 x 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def to_tensor(images):
    """uint8 (N, 28, 28) -> float32 (N, 1, 28, 28) в диапазоне 0..1."""
    return torch.from_numpy(np.asarray(images, dtype=np.float32) / 255.0).unsqueeze(1)


def save_model_to_bucket(s3, model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    s3.put_object(Bucket=bucket(), Key=MODEL_KEY, Body=buf.getvalue())


def load_model_from_bucket(s3):
    obj = s3.get_object(Bucket=bucket(), Key=MODEL_KEY)
    model = FashionCNN()
    model.load_state_dict(torch.load(io.BytesIO(obj["Body"].read()), map_location="cpu"))
    model.eval()
    return model
