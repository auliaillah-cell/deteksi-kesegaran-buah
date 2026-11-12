"""Ambil sampel foto per jenis dari Fruits-360 ke folder tanpa label kesegaran.

Sumber: https://github.com/fruits-360/fruits-360-100x100
Lisensi sumber: CC BY-SA 4.0. Tidak memasukkan gambar ke kategori segar/busuk.
"""
import concurrent.futures
import csv
import json
import urllib.request
from pathlib import Path

from common import DATASET, FRUITS

MAX_PER_FRUIT = 100
DESTINATION = "belum_berlabel_kesegaran"
API_ROOT = "https://api.github.com/repos/fruits-360/fruits-360-100x100/contents/Training?ref=main"

# Nama direktori pada sumber -> ID buah project. Beberapa jenis memiliki banyak varietas.
SOURCE_PREFIXES = {
    "almond": ("almond",), "apple": ("apple",), "apricot": ("apricot",),
    "avocado": ("avocado",), "banana": ("banana",), "blackberry": ("blackberry",),
    "blueberry": ("blueberry",), "cantaloupe": ("cantaloupe",), "cherry": ("cherry",),
    "clementine": ("clementine",), "coconut": ("cocos",), "date": ("dates",),
    "fig": ("fig",), "grape": ("grape",), "kiwi_fruit": ("kiwi",),
    "lemon": ("lemon",), "lime": ("lime",), "mandarin_orange": ("mandarine",),
    "melon": ("melon",), "orange": ("orange",), "papaya": ("papaya",),
    "peach": ("peach",), "pear": ("pear",), "persimmon": ("kaki",),
    "pineapple": ("pineapple",), "raspberry": ("raspberry",),
    "strawberry": ("strawberry",), "tomato": ("tomato",), "watermelon": ("watermelon",),
    # prune sengaja tidak dipetakan ke plum: padanan itu belum terverifikasi.
}

def main():
    def get_json(url):
        request = urllib.request.Request(url, headers={"User-Agent": "fruit-dataset-class-project"})
        with urllib.request.urlopen(request, timeout=45) as response:
            return json.load(response)

    print("Membaca daftar kategori Fruits-360 dari GitHub...")
    source_dirs = get_json(API_ROOT)
    selected = {}
    # Ambil satu direktori varietas representatif per jenis agar unduhan tidak
    # menarik seluruh arsip 200 ribu foto. Foto sumber tetap hanya foto jenis.
    for fruit, prefixes in SOURCE_PREFIXES.items():
        match = next((entry for entry in source_dirs if entry["type"] == "dir" and
                      any(entry["name"].lower().startswith(prefix) for prefix in prefixes)), None)
        if match:
            selected[fruit] = match
    output_manifest = DATASET / "sumber_fruits360.csv"
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    copied, absent = 0, []
    with output_manifest.open("w", newline="", encoding="utf-8-sig") as manifest_file:
        writer = csv.writer(manifest_file)
        writer.writerow(("fruit_id", "file", "source_class", "source_split", "source_url", "license", "freshness_label"))
        for fruit, _label in FRUITS:
            folder = selected.get(fruit)
            if not folder:
                absent.append(fruit)
                continue
            entries = [entry for entry in get_json(folder["url"]) if entry["type"] == "file" and
                       entry["name"].lower().endswith(".jpg")]
            entries.sort(key=lambda entry: entry["name"])
            if len(entries) > MAX_PER_FRUIT:
                indices = [int(i * len(entries) / MAX_PER_FRUIT) for i in range(MAX_PER_FRUIT)]
                entries = [entries[i] for i in indices]
            destination = DATASET / fruit / DESTINATION
            destination.mkdir(parents=True, exist_ok=True)
            def fetch(entry):
                target = destination / entry["name"]
                if target.exists() and target.stat().st_size:
                    return (entry, target, "already_present")
                request = urllib.request.Request(entry["download_url"], headers={"User-Agent": "fruit-dataset-class-project"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    payload = response.read()
                target.write_bytes(payload)
                return (entry, target, "downloaded")

            # Unduhan paralel mempercepat permintaan file kecil tanpa mengunduh
            # repo penuh. Manifest diperbarui hanya untuk foto yang berhasil.
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(fetch, entry) for entry in entries]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        entry, target, _status = future.result()
                    except Exception as error:
                        print(f"Gagal mengunduh {fruit}: {error}")
                        continue
                    writer.writerow((fruit, target.relative_to(DATASET).as_posix(), folder["name"],
                                     "Training", "https://github.com/fruits-360/fruits-360-100x100",
                                     "CC BY-SA 4.0", ""))
                    copied += 1
            print(f"{fruit}: {len(entries)} foto diproses")
    print(f"Foto disalin: {copied}; buah tanpa foto sumber: {', '.join(absent) or 'tidak ada'}")
    print(f"Prune dilewati karena tidak ada kelas prune yang sepadan; manifest: {output_manifest}")
    print("Foto berada di folder tanpa label kesegaran dan tidak dihitung sebagai segar/busuk.")

if __name__ == "__main__":
    main()
