"""Шаг 1. Скачивает Fashion-MNIST из официального репозитория и загружает в бакет."""
import urllib.request

from common import DATA_PREFIX, FILES, bucket, s3_client

BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"


def main():
    s3 = s3_client()
    for name, filename in FILES.items():
        print(f"скачиваю {filename} ...", end=" ", flush=True)
        raw = urllib.request.urlopen(BASE_URL + filename, timeout=120).read()
        s3.put_object(Bucket=bucket(), Key=DATA_PREFIX + filename, Body=raw)
        print(f"{len(raw) / 1e6:.1f} МБ -> s3://{bucket()}/{DATA_PREFIX}{filename}")

    print("\nСодержимое бакета:")
    for obj in s3.list_objects_v2(Bucket=bucket(), Prefix=DATA_PREFIX).get("Contents", []):
        print(f"  {obj['Key']:55s} {obj['Size']:>10d} байт")


if __name__ == "__main__":
    main()
