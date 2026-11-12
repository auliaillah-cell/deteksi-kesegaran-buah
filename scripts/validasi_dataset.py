import csv
import cv2
from common import DATASET, ROOT, FRUITS, FRESHNESS, image_files

def main():
    errors, rows = [], []
    for fruit, label in FRUITS:
        for condition in FRESHNESS:
            folder = DATASET / fruit / condition
            folder.mkdir(parents=True, exist_ok=True)
            count = 0
            for path in image_files(folder):
                image = cv2.imread(str(path))
                if image is None:
                    errors.append(f"Gambar rusak/tidak terbaca: {path.relative_to(ROOT)}")
                else:
                    count += 1
            rows.append((fruit, label, condition, count))
        count_unlabeled = 0
        for path in image_files(DATASET / fruit / "belum_berlabel_kesegaran"):
            image = cv2.imread(str(path))
            if image is None:
                errors.append(f"Gambar rusak/tidak terbaca: {path.relative_to(ROOT)}")
            else:
                count_unlabeled += 1
        rows.append((fruit, label, "belum_berlabel_kesegaran", count_unlabeled))
    report = ROOT / "hasil_validasi.csv"
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(("fruit", "nama", "condition", "gambar_valid")); w.writerows(rows)
    (ROOT / "gambar_rusak.txt").write_text("\n".join(errors), encoding="utf-8")
    print(f"Validasi selesai. {len(errors)} gambar bermasalah; rincian: gambar_rusak.txt")

if __name__ == "__main__":
    main()
