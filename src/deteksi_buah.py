
import hashlib
from pathlib import Path
from urllib.request import urlopen

import cv2
from ultralytics import YOLO


KELAS_BUAH = {
    "almond": "Almond",
    "apple": "Apel",
    "apricot": "Aprikot",
    "avocado": "Alpukat",
    "banana": "Pisang",
    "blackberry": "Blackberry",
    "blueberry": "Blueberry",
    "cantaloup/cantaloupe": "Melon",
    "cherry": "Ceri",
    "clementine": "Clementine",
    "coconut/cocoanut": "Kelapa",
    "date/date fruit": "Kurma",
    "fig/fig fruit": "Ara",
    "grape": "Anggur",
    "kiwi fruit": "Kiwi",
    "lemon": "Lemon",
    "lime": "Jeruk nipis",
    "mandarin orange": "Jeruk mandarin",
    "melon": "Melon",
    "orange/orange fruit": "Jeruk",
    "papaya": "Pepaya",
    "peach": "Persik",
    "pear": "Pir",
    "persimmon": "Kesemek",
    "pineapple": "Nanas",
    "prune": "Plum",
    "raspberry": "Raspberry",
    "strawberry": "Stroberi",
    "tomato": "Tomat",
    "watermelon": "Semangka",
    
}

MODEL_URL = (
    "https://huggingface.co/Senu-12/snapstock-fruit-vegetable-detector/"
    "resolve/main/yolov8/fruit_vegetable_yolov8m.pt"
)
MODEL_SHA256 = "a18624d446e2fea08384949115de0b49446f0be2487fc954a37a9d962f71e5d3"
MODEL_SIZE = 52_089_985
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "model"
    / "fruit_vegetable_yolov8m.pt"
)
MODEL_JERUK_PATH = Path(__file__).resolve().parent.parent / "yolo26n.pt"

model = None
model_jeruk = None


def _sha256_file(path):
    sha256 = hashlib.sha256()
    with path.open("rb") as file_model:
        for potongan in iter(lambda: file_model.read(1024 * 1024), b""):
            sha256.update(potongan)
    return sha256.hexdigest()


def _unduh_model(lokasi_model):
    lokasi_model.parent.mkdir(parents=True, exist_ok=True)
    lokasi_sementara = lokasi_model.with_suffix(".pt.part")

    try:
        with urlopen(MODEL_URL, timeout=60) as respons:
            with lokasi_sementara.open("wb") as file_model:
                while True:
                    potongan = respons.read(1024 * 1024)
                    if not potongan:
                        break
                    file_model.write(potongan)

        if lokasi_sementara.stat().st_size != MODEL_SIZE:
            raise RuntimeError("Ukuran file model yang diunduh tidak sesuai.")

        if _sha256_file(lokasi_sementara) != MODEL_SHA256:
            raise RuntimeError("Verifikasi keamanan file model gagal.")

        lokasi_sementara.replace(lokasi_model)
    except Exception as error:
        lokasi_sementara.unlink(missing_ok=True)
        raise RuntimeError(
            "Model buah gagal disiapkan. Periksa koneksi internet lalu coba lagi. "
            f"Detail: {error}"
        ) from error


def _dapatkan_model():
    global model

    if model is None:
        model_valid = (
            MODEL_PATH.is_file()
            and MODEL_PATH.stat().st_size == MODEL_SIZE
            and _sha256_file(MODEL_PATH) == MODEL_SHA256
        )
        if not model_valid:
            _unduh_model(MODEL_PATH)

        model_baru = YOLO(str(MODEL_PATH))
        nama_kelas = {
            nama.strip().lower(): nomor
            for nomor, nama in model_baru.names.items()
        }
        kelas_tidak_ditemukan = set(KELAS_BUAH) - set(nama_kelas)
        if kelas_tidak_ditemukan:
            raise RuntimeError(
                "Model tidak memiliki semua kelas buah yang dibutuhkan: "
                + ", ".join(sorted(kelas_tidak_ditemukan))
            )

        model = model_baru

    return model


def _dapatkan_model_jeruk():
    global model_jeruk

    if model_jeruk is None:
        if not MODEL_JERUK_PATH.is_file():
            raise FileNotFoundError(
                f"Model pendeteksi jeruk tidak ditemukan: {MODEL_JERUK_PATH}"
            )

        model_baru = YOLO(str(MODEL_JERUK_PATH))
        if not any(
            nama.strip().lower() == "orange"
            for nama in model_baru.names.values()
        ):
            raise RuntimeError("Model pendeteksi jeruk tidak memiliki kelas orange.")

        model_jeruk = model_baru

    return model_jeruk


def _iou(kotak_satu, kotak_dua):
    x1 = max(kotak_satu[0], kotak_dua[0])
    y1 = max(kotak_satu[1], kotak_dua[1])
    x2 = min(kotak_satu[2], kotak_dua[2])
    y2 = min(kotak_satu[3], kotak_dua[3])
    lebar_irisan = max(0, x2 - x1)
    tinggi_irisan = max(0, y2 - y1)
    luas_irisan = lebar_irisan * tinggi_irisan
    luas_satu = max(0, kotak_satu[2] - kotak_satu[0]) * max(
        0, kotak_satu[3] - kotak_satu[1]
    )
    luas_dua = max(0, kotak_dua[2] - kotak_dua[0]) * max(
        0, kotak_dua[3] - kotak_dua[1]
    )
    luas_gabungan = luas_satu + luas_dua - luas_irisan
    return luas_irisan / luas_gabungan if luas_gabungan else 0


def _id_kelas_buah(nama_kelas):
    return [
        nomor
        for nomor, nama in nama_kelas.items()
        if nama.strip().lower() in KELAS_BUAH
    ]


def _tambahkan_deteksi_unik(daftar_buah, buah):
    for indeks, buah_lama in enumerate(daftar_buah):
        if (
            buah_lama["nama"] == buah["nama"]
            and _iou(buah_lama["koordinat"], buah["koordinat"]) >= 0.5
        ):
            if (
                buah["keyakinan_deteksi"]
                > buah_lama["keyakinan_deteksi"]
            ):
                daftar_buah[indeks] = buah
            return

    daftar_buah.append(buah)


def deteksi_buah(lokasi_foto):
    """
    Mendeteksi buah pada foto dan menampilkan
    kotak serta label yang lebih mudah dibaca.
    """

    # Memastikan foto bisa dibaca
    gambar = cv2.imread(lokasi_foto)

    if gambar is None:
        raise ValueError("Foto tidak ditemukan atau tidak bisa dibaca.")

    model_deteksi = _dapatkan_model()
    id_buah = _id_kelas_buah(model_deteksi.names)
    hasil = model_deteksi.predict(
        source=gambar,
        classes=id_buah,
        conf=0.15,
        iou=0.45,
        imgsz=640,
        max_det=100,
        device="cpu",
        verbose=False
    )

    daftar_buah = []

    for prediksi in hasil:
        for kotak in prediksi.boxes:
            nomor_kelas = int(kotak.cls[0])
            nama_inggris = model_deteksi.names[nomor_kelas].strip().lower()

            if nama_inggris not in KELAS_BUAH:
                continue

            koordinat = (
                kotak.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
            )

            keyakinan = float(kotak.conf[0])

            _tambahkan_deteksi_unik(daftar_buah, {
                "nama": KELAS_BUAH[nama_inggris],
                "koordinat": koordinat.tolist(),
                "keyakinan_deteksi": keyakinan
            })

    model_jeruk_deteksi = _dapatkan_model_jeruk()
    id_jeruk = [
        nomor
        for nomor, nama in model_jeruk_deteksi.names.items()
        if nama.strip().lower() == "orange"
    ]
    hasil_jeruk = model_jeruk_deteksi.predict(
        source=gambar,
        classes=id_jeruk,
        conf=0.20,
        iou=0.45,
        imgsz=640,
        max_det=100,
        device="cpu",
        verbose=False
    )

    for prediksi in hasil_jeruk:
        for kotak in prediksi.boxes:
            koordinat = (
                kotak.xyxy[0]
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )
            daftar_buah = [
                buah
                for buah in daftar_buah
                if _iou(buah["koordinat"], koordinat) < 0.5
            ]
            _tambahkan_deteksi_unik(daftar_buah, {
                "nama": "Jeruk",
                "koordinat": koordinat,
                "keyakinan_deteksi": float(kotak.conf[0])
            })

    # Menggambar kotak dan label yang lebih rapi
    for nomor, buah in enumerate(daftar_buah, start=1):
        x1, y1, x2, y2 = buah["koordinat"]

        # Kotak hijau yang lebih tipis
        cv2.rectangle(
            gambar,
            (x1, y1),
            (x2, y2),
            (0, 200, 0),
            2
        )

        # Label singkat dengan ukuran lebih kecil
        label = (
            f"{nomor}. {buah['nama']} "
            f"{buah['keyakinan_deteksi']:.0%}"
        )

        ukuran_teks, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )

        lebar_teks, tinggi_teks = ukuran_teks

        # Label diletakkan di atas kotak jika memungkinkan
        label_y = max(y1, tinggi_teks + 10)

        # Latar belakang hijau agar tulisan jelas
        cv2.rectangle(
            gambar,
            (x1, label_y - tinggi_teks - 8),
            (x1 + lebar_teks + 8, label_y + baseline - 4),
            (0, 160, 0),
            -1
        )

        # Tulisan putih agar kontras
        cv2.putText(
            gambar,
            label,
            (x1 + 4, label_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return gambar, daftar_buah
