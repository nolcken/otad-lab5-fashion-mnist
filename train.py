"""Шаг 2. Обучение CNN на облачной ВМ: данные читаются из бакета, модель и метрики пишутся в бакет."""
import argparse
import io
import json
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import (CLASSES, METRICS_KEY, MODEL_KEY, FashionCNN, bucket, load_from_bucket,
                    s3_client, save_model_to_bucket, to_tensor)


def evaluate(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(model(x).argmax(1))
    return torch.cat(preds).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(torch.get_num_threads())

    s3 = s3_client()
    t0 = time.time()
    x_train, y_train = load_from_bucket(s3, "train_images"), load_from_bucket(s3, "train_labels")
    x_test, y_test = load_from_bucket(s3, "test_images"), load_from_bucket(s3, "test_labels")
    print(f"данные из s3://{bucket()}: train {x_train.shape}, test {x_test.shape}, {time.time() - t0:.1f} с")

    train_loader = DataLoader(TensorDataset(to_tensor(x_train), torch.from_numpy(y_train.astype(np.int64))),
                              batch_size=args.batch, shuffle=True)
    test_loader = DataLoader(TensorDataset(to_tensor(x_test), torch.from_numpy(y_test.astype(np.int64))),
                             batch_size=1000)

    model = FashionCNN()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"параметров модели: {sum(p.numel() for p in model.parameters()):,}, потоков CPU: {torch.get_num_threads()}")

    history = []
    t_train = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep, total, n = time.time(), 0.0, 0
        for x, y in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
            n += len(y)
        acc = float((evaluate(model, test_loader) == y_test).mean())
        history.append({"epoch": epoch, "loss": round(total / n, 4), "test_accuracy": round(acc, 4),
                        "seconds": round(time.time() - t_ep, 1)})
        print(f"эпоха {epoch}: loss={total / n:.4f}, accuracy(test)={acc:.4f}, {time.time() - t_ep:.0f} с")
    train_seconds = time.time() - t_train

    pred = evaluate(model, test_loader)
    accuracy = float((pred == y_test).mean())
    per_class = {CLASSES[c]: round(float((pred[y_test == c] == c).mean()), 4) for c in range(10)}
    confusion = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, pred):
        confusion[t, p] += 1

    print(f"\nитоговая accuracy на тесте: {accuracy:.4f} (критерий >= 0.88: {'выполнен' if accuracy >= 0.88 else 'НЕ выполнен'})")
    print("accuracy по классам:")
    for name, value in per_class.items():
        print(f"  {name:12s} {value:.4f}")

    save_model_to_bucket(s3, model)
    metrics = {"accuracy": round(accuracy, 4), "per_class": per_class, "confusion": confusion.tolist(),
               "history": history, "train_seconds": round(train_seconds, 1), "epochs": args.epochs,
               "batch": args.batch, "lr": args.lr}
    s3.put_object(Bucket=bucket(), Key=METRICS_KEY, Body=json.dumps(metrics, ensure_ascii=False, indent=2).encode())
    with open("metrics_v1.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nмодель: s3://{bucket()}/{MODEL_KEY}\nметрики: s3://{bucket()}/{METRICS_KEY}")
    print(f"время обучения: {train_seconds:.0f} с")


if __name__ == "__main__":
    main()
