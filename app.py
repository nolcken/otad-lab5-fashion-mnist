"""Шаг 3. REST-сервис: при старте берёт модель из бакета, POST /predict возвращает класс и вероятность."""
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import json

from common import CLASSES, METRICS_KEY, MODEL_KEY, bucket, load_model_from_bucket, s3_client, to_tensor

app = FastAPI(title="Fashion-MNIST CNN", version="1.0")
model = None
metrics = {}


class Image(BaseModel):
    pixels: List[List[float]]   # 28 строк по 28 значений яркости 0..255


@app.on_event("startup")
def load_model():
    global model, metrics
    torch.set_num_threads(1)
    s3 = s3_client()
    model = load_model_from_bucket(s3)
    metrics = json.loads(s3.get_object(Bucket=bucket(), Key=METRICS_KEY)["Body"].read())
    print(f"модель загружена из s3://{bucket()}/{MODEL_KEY}")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_KEY, "loaded": model is not None}


@app.get("/metrics")
def model_metrics():
    """Метрики качества, сохранённые в бакет при обучении."""
    return {k: metrics.get(k) for k in ("accuracy", "per_class", "history", "train_seconds", "epochs")}


@app.post("/predict")
def predict(img: Image):
    arr = np.asarray(img.pixels, dtype=np.float32)
    if arr.shape != (28, 28):
        raise HTTPException(422, f"ожидается массив 28x28, получено {list(arr.shape)}")
    if arr.min() < 0 or arr.max() > 255:
        raise HTTPException(422, "значения пикселей должны быть от 0 до 255")
    with torch.no_grad():
        prob = torch.softmax(model(to_tensor(arr[None])), dim=1)[0]
    k = int(prob.argmax())
    return {"class_id": k, "class_name": CLASSES[k], "probability": round(float(prob[k]), 4)}
