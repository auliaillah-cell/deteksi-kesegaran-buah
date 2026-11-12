"""Bagi terstratifikasi per buah/kesegaran. Bersihkan output split lama sebelum menjalankan ulang."""
import hashlib
import shutil
from common import DATASET, SPLIT_ROOT, FRUITS, FRESHNESS, SPLITS, image_files

def main():
    counts = {name: 0 for name, _ in SPLITS}
    for fruit, _ in FRUITS:
        for condition in FRESHNESS:
            files = image_files(DATASET / fruit / condition)
            files.sort(key=lambda p: hashlib.sha256(str(p.resolve()).encode()).hexdigest())
            n = len(files)
            train_n = int(n * .70)
            val_n = int(n * .20)
            groups = (("train", files[:train_n]), ("val", files[train_n:train_n + val_n]),
                      ("test", files[train_n + val_n:]))
            for split, members in groups:
                dest = SPLIT_ROOT / split / fruit / condition
                dest.mkdir(parents=True, exist_ok=True)
                for path in members:
                    target = dest / path.name
                    if target.exists():
                        target = dest / f"{path.stem}_{hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]}{path.suffix}"
                    shutil.copy2(path, target)
                    counts[split] += 1
    print("Split selesai:", counts, "(pembulatan dilakukan per kategori)")

if __name__ == "__main__":
    main()
