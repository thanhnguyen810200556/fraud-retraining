import csv
import hashlib
from pathlib import Path
from datetime import date

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = RAW_DIR / "dataset_manifest.csv"

# Metadata cần xác nhận thủ công từ nguồn tải
DATASET_METADATA = {
    "ieee_cis": {
        "dataset_name": "IEEE-CIS Fraud Detection",
        "dataset_role": "Main",
        "source_url": (
            "https://www.kaggle.com/competitions/" "ieee-fraud-detection/data"
        ),
        "version": "Unverified",
        "license": "Kaggle Competition Rules",
        "download_date": "2026-11-24",
    },
    "baf": {
        "dataset_name": "Bank Account Fraud",
        "dataset_role": "Secondary",
        "source_url": (
            "https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022"
        ),
        "version": "Unverified",
        "license": "To be verified",
        "download_date": "2026-11-24",
    },
}


def calculate_sha256(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_csv_shape(file_path):
    with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)

        header = next(reader, [])
        num_columns = len(header)
        num_rows = sum(1 for _ in reader)

    return num_rows, num_columns


def generate_manifest():
    records = []

    for folder, metadata in DATASET_METADATA.items():
        dataset_dir = RAW_DIR / folder

        if not dataset_dir.exists():
            continue

        for file_path in sorted(dataset_dir.rglob("*.csv")):
            print(f"Processing: {file_path.name}")

            num_rows, num_columns = get_csv_shape(file_path)

            record = {
                **metadata,
                "file_name": file_path.name,
                "file_size_mb": round(file_path.stat().st_size / 1024**2, 2),
                "num_rows": num_rows,
                "num_columns": num_columns,
                "sha256": calculate_sha256(file_path),
                "local_path": file_path.relative_to(PROJECT_ROOT).as_posix(),
                "recorded_date": date.today().isoformat(),
                "notes": "",
            }

            records.append(record)

    if not records:
        print("No dataset files found.")
        return

    with open(OUTPUT_FILE, "w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    print(f"Manifest saved to: {OUTPUT_FILE}")
    print(f"Total files: {len(records)}")


if __name__ == "__main__":
    generate_manifest()
