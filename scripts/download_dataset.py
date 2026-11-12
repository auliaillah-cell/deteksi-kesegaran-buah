"""Impor gambar lokal yang sudah dikurasi; tidak mengunduh atau memberi label otomatis.

Isi dataset_manifest.csv dengan source_path,fruit,condition,source_url,license,attribution.
Setiap baris harus menunjuk gambar yang memang sudah diverifikasi manusia.
"""
import csv
import shutil
from pathlib import Path
from common import ROOT, DATASET, FRUITS, FRESHNESS, IMAGE_EXTS

MANIFEST = ROOT / "dataset_manifest.csv"
FIELDS = ("source_path", "fruit", "condition", "source_url", "license", "attribution")

def main():
    MANIFEST.touch(exist_ok=True)
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise SystemExit("Manifest harus memiliki kolom: " + ",".join(FIELDS))
        rows = list(reader)
    allowed_fruits = {key for key, _ in FRUITS}
    copied = 0
    for number, row in enumerate(rows, start=2):
        source = Path(row["source_path"].strip()).expanduser()
        if not source.is_absolute():
            source = ROOT / source
        fruit, condition = row["fruit"].strip(), row["condition"].strip()
        if fruit not in allowed_fruits or condition not in FRESHNESS:
            print(f"Lewati baris {number}: nama buah/kategori tidak valid")
            continue
        if source.suffix.lower() not in IMAGE_EXTS or not source.is_file():
            print(f"Lewati baris {number}: berkas gambar tidak ditemukan/format tidak didukung")
            continue
        if not all(row.get(k, "").strip() for k in ("source_url", "license")):
            print(f"Lewati baris {number}: sumber URL dan lisensi wajib dicatat")
            continue
        target_dir = DATASET / fruit / condition
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / source.name
        if target.exists():
            target = target_dir / f"{source.stem}_{number}{source.suffix.lower()}"
        shutil.copy2(source, target)
        copied += 1
    print(f"Impor selesai: {copied} gambar disalin. Pastikan label dan lisensi sudah ditinjau.")

if __name__ == "__main__":
    main()
