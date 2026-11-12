import hashlib
import csv
from common import DATASET, FRUITS, image_files

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    seen, removed, removed_paths = {}, 0, set()
    for fruit, _ in FRUITS:
        for path in image_files(DATASET / fruit):
            key = digest(path)
            if key in seen:
                path.unlink()
                removed += 1
                removed_paths.add(path.relative_to(DATASET).as_posix())
                print(f"Duplikat dihapus: {path} (sama dengan {seen[key]})")
            else:
                seen[key] = path
    manifest = DATASET / "sumber_fruits360.csv"
    if removed_paths and manifest.is_file():
        with manifest.open("r", newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            fields, rows = reader.fieldnames, list(reader)
        with manifest.open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row["file"] not in removed_paths)
    print(f"Selesai: {removed} duplikat identik dihapus.")

if __name__ == "__main__":
    main()
