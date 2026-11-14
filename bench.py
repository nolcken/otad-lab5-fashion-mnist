"""Шаг 4. 100 запросов к сервису: среднее время ответа и p95, плюс проверка ответов на тестовых картинках."""
import argparse
import gzip
import statistics
import time
import urllib.request

import numpy as np
import requests

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"


def load_test(n):
    imgs = gzip.decompress(urllib.request.urlopen(BASE_URL + "t10k-images-idx3-ubyte.gz", timeout=120).read())
    labels = gzip.decompress(urllib.request.urlopen(BASE_URL + "t10k-labels-idx1-ubyte.gz", timeout=120).read())
    x = np.frombuffer(imgs, dtype=np.uint8, offset=16).reshape(-1, 28, 28)[:n]
    y = np.frombuffer(labels, dtype=np.uint8, offset=8)[:n]
    return x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="адрес сервиса, например http://84.201.1.2:8000")
    ap.add_argument("-n", type=int, default=100)
    args = ap.parse_args()

    x, y = load_test(args.n)
    session = requests.Session()
    session.post(f"{args.url}/predict", json={"pixels": x[0].tolist()}, timeout=30)   # прогрев

    times, correct = [], 0
    for img, label in zip(x, y):
        t0 = time.perf_counter()
        r = session.post(f"{args.url}/predict", json={"pixels": img.tolist()}, timeout=30)
        times.append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        correct += int(r.json()["class_id"] == label)

    times_sorted = sorted(times)
    p95 = times_sorted[int(round(0.95 * len(times))) - 1]
    print(f"запросов: {len(times)}, верных ответов: {correct}")
    print(f"среднее: {statistics.mean(times):.1f} мс, медиана: {statistics.median(times):.1f} мс, "
          f"p95: {p95:.1f} мс, min: {times_sorted[0]:.1f} мс, max: {times_sorted[-1]:.1f} мс")
    print("пример ответа:", r.json())


if __name__ == "__main__":
    main()
