import csv
from common import DATASET, ROOT, FRUITS, FRESHNESS, image_files

def main():
    rows = []
    for index, (fruit, label) in enumerate(FRUITS, 1):
        counts = [len(image_files(DATASET / fruit / category)) for category in FRESHNESS]
        unlabelled = len(image_files(DATASET / fruit / "belum_berlabel_kesegaran"))
        rows.append((index, fruit, label, *counts, sum(counts), unlabelled))
    with (ROOT / "laporan_dataset.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(("No", "ID Buah", "Jenis Buah", *FRESHNESS, "Total Berlabel", "Belum berlabel kesegaran")); w.writerows(rows)
    lines = ["# Laporan Dataset", "", "Jumlah dihitung dari gambar yang tersedia pada folder sumber, bukan target.",
             "Foto tanpa label kesegaran dilaporkan terpisah dan tidak masuk total berlabel.",
             "", "| No | Jenis Buah | Segar | Kurang Segar | Busuk | Total Berlabel | Belum Berlabel Kesegaran |", "|---:|---|---:|---:|---:|---:|---:|"]
    lines.extend(f"| {i} | {label} | {a} | {b} | {c} | {total} | {unlabelled} |" for i, _, label, a, b, c, total, unlabelled in rows)
    lines.append(f"|  | **Total** | **{sum(r[3] for r in rows)}** | **{sum(r[4] for r in rows)}** | **{sum(r[5] for r in rows)}** | **{sum(r[6] for r in rows)}** | **{sum(r[7] for r in rows)}** |")
    (ROOT / "laporan_dataset.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Laporan dibuat: laporan_dataset.csv dan laporan_dataset.md")

if __name__ == "__main__":
    main()
