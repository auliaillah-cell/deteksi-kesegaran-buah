
import hashlib
from pathlib import Path
from urllib.request import urlopen

try:
    import cv2
except ImportError:
    cv2 = None
    from PIL import Image
    import colorsys

import numpy as np
import colorsys  # for HSV conversion

# New helper: compute average RGB and HSV for a ROI (BGR image)
def _rata_rgb_hsv(potongan_bgr):
    """Return average RGB (0‑255) and HSV (H 0‑179, S/V 0‑255) for the given BGR ROI.
    The ROI may be empty; in that case return zeros.
    """
    if potongan_bgr is None or potongan_bgr.size == 0:
        return (0, 0, 0), (0, 0, 0)
    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(potongan_bgr, cv2.COLOR_BGR2RGB)
    # Average per channel
    avg_rgb = tuple(int(roi_rgb[:, :, i].mean()) for i in range(3))
    # Convert to HSV (OpenCV scale: H 0‑179, S/V 0‑255)
    roi_hsv = cv2.cvtColor(potongan_bgr, cv2.COLOR_BGR2HSV)
    avg_hsv = tuple(int(roi_hsv[:, :, i].mean()) for i in range(3))
    return avg_rgb, avg_hsv

from ultralytics import YOLO
# Dukung pemanggilan sebagai modul (`python -m src.deteksi_buah`) maupun
# langsung (`python src/deteksi_buah.py`).
if __package__:
    from .kesegaran import klasifikasikan_kesegaran
else:
    from kesegaran import klasifikasikan_kesegaran


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
CONFIDENCE_THRESHOLD = 0.15
IOU_THRESHOLD = 0.45
IOU_DUPLIKAT_LINTAS_KELAS = 0.70
IMAGE_SIZE = 640
MAX_DETECTIONS = 100
NAMA_BUAH_TIDAK_TERIDENTIFIKASI = "Buah tidak teridentifikasi"
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


def _gabungkan_deteksi_jeruk(daftar_buah, deteksi_jeruk):
    """Gabungkan model jeruk tanpa menimpa kelas buah lain yang bertumpang tindih."""
    indeks_beririsan = [
        indeks
        for indeks, buah_lama in enumerate(daftar_buah)
        if _iou(buah_lama["koordinat"], deteksi_jeruk["koordinat"]) >= 0.5
    ]

    if not indeks_beririsan:
        _tambahkan_deteksi_unik(daftar_buah, deteksi_jeruk)
        return

    # Model kedua hanya boleh memperbarui hasil citrus. Jangan menimpa kelas
    # lain atau menggabungkan beberapa objek yang kotaknya berbeda.
    if any(
        daftar_buah[indeks]["nama"] not in {"Jeruk", "Jeruk mandarin"}
        for indeks in indeks_beririsan
    ):
        return

    if any(
        _iou(
            daftar_buah[indeks_satu]["koordinat"],
            daftar_buah[indeks_dua]["koordinat"],
        ) < IOU_DUPLIKAT_LINTAS_KELAS
        for posisi, indeks_satu in enumerate(indeks_beririsan)
        for indeks_dua in indeks_beririsan[posisi + 1:]
    ):
        return

    indeks_sisip = min(indeks_beririsan)
    for indeks in sorted(indeks_beririsan, reverse=True):
        del daftar_buah[indeks]
    daftar_buah.insert(indeks_sisip, deteksi_jeruk)


# Helper to count fruits per freshness category
from collections import Counter

def _hitung_total_buah(daftar_buah):
    """Return a dict with counts per freshness (segar, kurang_segar, busuk) and total.
    Buah yang tidak memiliki kunci 'kesegaran' diabaikan dalam kategori.
    """
    counter = Counter()
    for buah in daftar_buah:
        kesegaran = buah.get('kesegaran')
        if kesegaran:
            counter[kesegaran] += 1
    total = sum(counter.values())
    result = {
        'segar': counter.get('segar', 0),
        'kurang_segar': counter.get('kurang_segar', 0),
        'busuk': counter.get('busuk', 0),
        'total': total,
    }
    return result

def _gabungkan_kotak_lintas_kelas(daftar_buah):
    """Hitung satu kotak yang bertumpuk lintas kelas dan tandai kelasnya ambigu."""
    daftar_terurut = sorted(
        daftar_buah,
        key=lambda buah: buah["keyakinan_deteksi"],
        reverse=True,
    )
    hasil = []

    for buah in daftar_terurut:
        indeks_beririsan = [
            indeks
            for indeks, buah_lama in enumerate(hasil)
            if _iou(buah_lama["koordinat"], buah["koordinat"])
            >= IOU_DUPLIKAT_LINTAS_KELAS
        ]

        # Satu kotak lebar yang menutupi beberapa hasil terpisah tidak boleh
        # menggabungkan buah-buah tersebut menjadi satu objek.
        if len(indeks_beririsan) > 1:
            continue
        if not indeks_beririsan:
            hasil.append(buah)
            continue

        indeks = indeks_beririsan[0]
        buah_lama = hasil[indeks]
        if buah_lama["nama"] == buah["nama"]:
            continue

        kelas_alternatif = set(buah_lama.get("kelas_alternatif", ()))
        kelas_alternatif.update(buah.get("kelas_alternatif", ()))
        if buah_lama["nama"] != NAMA_BUAH_TIDAK_TERIDENTIFIKASI:
            kelas_alternatif.add(buah_lama["nama"])
        if buah["nama"] != NAMA_BUAH_TIDAK_TERIDENTIFIKASI:
            kelas_alternatif.add(buah["nama"])

        buah_ambigu = dict(buah_lama)
        buah_ambigu["nama"] = NAMA_BUAH_TIDAK_TERIDENTIFIKASI
        buah_ambigu["kelas_alternatif"] = sorted(kelas_alternatif)
        hasil[indeks] = buah_ambigu

    return hasil


def deteksi_buah(lokasi_foto):
    """Detect fruits in an image and return annotated image + list of detections.
    Added debugging output for thresholds, image size, and raw prediction count.
    """
    # Debug: show configuration values
    print(f"[DEBUG] CONFIDENCE_THRESHOLD = {CONFIDENCE_THRESHOLD}", flush=True)
    print(f"[DEBUG] IOU_THRESHOLD = {IOU_THRESHOLD}", flush=True)
    print(f"[DEBUG] IMAGE_SIZE = {IMAGE_SIZE}", flush=True)
    print(f"[DEBUG] MAX_DETECTIONS = {MAX_DETECTIONS}", flush=True)
    # Ensure the image can be read
    gambar = cv2.imread(lokasi_foto)
    if gambar is None:
        raise ValueError("Foto tidak ditemukan atau tidak bisa dibaca.")
    # Debug: original image dimensions
    tinggi, lebar = gambar.shape[:2]
    print(f"[DEBUG] Image dimensions (lebar x tinggi): {lebar} x {tinggi}", flush=True)
    model_deteksi = _dapatkan_model()
    id_buah = _id_kelas_buah(model_deteksi.names)
    hasil = model_deteksi.predict(
        source=gambar,
        classes=id_buah,
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
        device="cpu",
        verbose=False,
    )
    # Debug: number of raw predictions (may include multiple classes)
    raw_counts = sum(len(p.boxes) for p in hasil)
    print(f"[DEBUG] Raw predictions (boxes) before filtering: {raw_counts}", flush=True)
    daftar_buah = []
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
        conf=CONFIDENCE_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMAGE_SIZE,
        max_det=MAX_DETECTIONS,
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
            x1, y1, x2, y2 = koordinat.tolist()
            potongan = gambar[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            kelas_segar, yakin_segar, indeks_segar = klasifikasikan_kesegaran(potongan)


            # New: compute average RGB and HSV for this ROI
            avg_rgb, avg_hsv = _rata_rgb_hsv(potongan)
            _tambahkan_deteksi_unik(daftar_buah, {
                "nama": KELAS_BUAH[nama_inggris],
                "koordinat": koordinat.tolist(),
                "keyakinan_deteksi": keyakinan,
                "kesegaran": kelas_segar,
                "keyakinan_kesegaran": yakin_segar,
                "indeks_kesegaran": indeks_segar,
                "rgb": avg_rgb,
                "hsv": avg_hsv,
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
            x1, y1, x2, y2 = koordinat
            potongan = gambar[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            kelas_segar, yakin_segar, indeks_segar = klasifikasikan_kesegaran(potongan)
            _gabungkan_deteksi_jeruk(daftar_buah, {
                "nama": "Jeruk",
                "koordinat": koordinat,
                "keyakinan_deteksi": float(kotak.conf[0]),
                "kesegaran": kelas_segar,
                "keyakinan_kesegaran": yakin_segar,
                "indeks_kesegaran": indeks_segar,
            })

    daftar_buah = _gabungkan_kotak_lintas_kelas(daftar_buah)

    # Cetak ringkasan citra dan semua hasil per buah ke terminal. Fungsi ini
    # dipanggil untuk setiap foto dari GUI maupun pemanggil langsung.
    tinggi, lebar = gambar.shape[:2]
    print(f"\nHasil deteksi: {Path(lokasi_foto).resolve()}", flush=True)
    print(f"Ukuran citra: {lebar} x {tinggi} piksel", flush=True)
    citra_grayscale = cv2.cvtColor(gambar, cv2.COLOR_BGR2GRAY)
    matriks_citra = cv2.resize(
        citra_grayscale, (12, 12), interpolation=cv2.INTER_AREA
    )
    print("Matriks intensitas grayscale (12 x 12, hasil resize):", flush=True)
    print(np.array2string(matriks_citra, separator=", ", max_line_width=120), flush=True)
    print(f"Jumlah buah terdeteksi: {len(daftar_buah)}", flush=True)
    if not daftar_buah:
        print("Tidak ada buah yang terdeteksi.", flush=True)
    for nomor, buah in enumerate(daftar_buah, start=1):
        x1, y1, x2, y2 = buah["koordinat"]
        print(
            f"{nomor}. {buah['nama']} | kondisi: "
            f"{buah['kesegaran'].replace('_', ' ')} | "
            f"indeks kesegaran: {buah['indeks_kesegaran']:.1f}% | "
            f"keyakinan kondisi: {buah['keyakinan_kesegaran']:.1%} | "
            f"keyakinan deteksi: {buah['keyakinan_deteksi']:.1%} | "
            f"kotak: x1={x1}, y1={y1}, x2={x2}, y2={y2}",
            flush=True,
        )

    # Menggambar kotak dan label yang lebih rapi
    for nomor, buah in enumerate(daftar_buah, start=1):
        x1, y1, x2, y2 = buah["koordinat"]

        warna = {
            "segar": (0, 180, 0),
            "kurang_segar": (0, 190, 230),
            "busuk": (0, 0, 220),
        }.get(buah["kesegaran"], (0, 180, 0))
        cv2.rectangle(
            gambar,
            (x1, y1),
            (x2, y2),
            warna,
            2
        )

        # Label singkat dengan ukuran lebih kecil
        label = (
            f"{nomor}. {buah['nama']} | {buah['kesegaran'].replace('_', ' ')} "
            f"{buah['indeks_kesegaran']:.0f}%"
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
            warna,
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
