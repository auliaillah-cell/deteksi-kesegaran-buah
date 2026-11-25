"""Model klasifikasi kesegaran buah per objek hasil deteksi."""

from pathlib import Path

import cv2
import torch
from torch import nn
from torchvision import transforms
from PIL import Image


KELAS_KESEGARAN = ("segar", "kurang_segar", "busuk")
MODEL_KESEGARAN_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "kesegaran_buah.pt"
)
UKURAN_INPUT = 160


class CNNKesegaran(nn.Module):
    """CNN kecil agar dapat dijalankan bersama detektor pada CPU."""

    def __init__(self, jumlah_kelas=len(KELAS_KESEGARAN)):
        super().__init__()
        lapisan = []
        kanal_masuk = 3
        for kanal_keluar in (24, 48, 96, 160):
            lapisan.extend((
                nn.Conv2d(kanal_masuk, kanal_keluar, 3, padding=1, bias=False),
                nn.BatchNorm2d(kanal_keluar),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ))
            kanal_masuk = kanal_keluar
        self.fitur = nn.Sequential(*lapisan, nn.AdaptiveAvgPool2d(1))
        self.kepala = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.25), nn.Linear(kanal_masuk, jumlah_kelas)
        )

    def forward(self, gambar):
        return self.kepala(self.fitur(gambar))


_model = None
_transformasi = transforms.Compose((
    transforms.Resize((UKURAN_INPUT, UKURAN_INPUT)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
))


def _dapatkan_model():
    global _model
    if _model is None:
        if not MODEL_KESEGARAN_PATH.is_file():
            raise FileNotFoundError(
                "Model kesegaran belum tersedia. Jalankan "
                ".venv\\Scripts\\python.exe scripts\\train_freshness.py"
            )
        data = torch.load(MODEL_KESEGARAN_PATH, map_location="cpu", weights_only=True)
        if tuple(data.get("kelas", ())) != KELAS_KESEGARAN:
            raise RuntimeError("Urutan kelas pada model kesegaran tidak cocok.")
        model = CNNKesegaran()
        model.load_state_dict(data["state_dict"])
        model.eval()
        _model = model
    return _model


def klasifikasikan_kesegaran(gambar_bgr):
    """Kembalikan kelas, keyakinan kelas, dan indeks kesegaran 0–100.
    Memperbaiki keputusan agar tidak mengklasifikasikan sebagai 'busuk' bila kepercayaan rendah.
    """
    if gambar_bgr is None or gambar_bgr.size == 0:
        raise ValueError("Potongan buah kosong dan tidak bisa diklasifikasikan.")
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transformasi(Image.fromarray(gambar_rgb)).unsqueeze(0)
    with torch.inference_mode():
        probabilitas_awal = torch.softmax(_dapatkan_model()(tensor), dim=1)[0]
    # Smoothing untuk mengurangi kepercayaan berlebih
    probabilitas = 0.9 * probabilitas_awal + 0.1 / len(KELAS_KESEGARAN)
    # Tentukan kelas dengan argmax, lalu koreksi bila 'busuk' dengan kepercayaan rendah
    indeks = int(torch.argmax(probabilitas))
    # Jika kelas 'busuk' terpilih tetapi keyakinannya di bawah ambang, pilih kelas berikutnya dengan keyakinan tertinggi
    if KELAS_KESEGARAN[indeks] == "busuk" and probabilitas[indeks] < 0.6:
        # Pilih kelas dengan probabilitas tertinggi selain 'busuk'
        mask = torch.tensor([k != "busuk" for k in KELAS_KESEGARAN])
        filtered = probabilitas * mask
        indeks = int(torch.argmax(filtered))
    # Hitung skor kesegaran ordinal (0‑100) berdasarkan probabilitas asli (tidak terfilter)
    nilai_kelas = torch.tensor((100.0, 50.0, 0.0))
    indeks_kesegaran = float(torch.sum(probabilitas * nilai_kelas))
    return (
        KELAS_KESEGARAN[indeks],
        float(probabilitas[indeks]),
        indeks_kesegaran,
    )
    """Kembalikan kelas, keyakinan kelas, dan indeks kesegaran 0–100.
    Memperbaiki logika pemilihan kelas agar tidak menglabeli buah sebagai 'busuk' saat kepercayaan rendah.
    """
    if gambar_bgr is None or gambar_bgr.size == 0:
        raise ValueError("Potongan buah kosong dan tidak bisa diklasifikasikan.")
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transformasi(Image.fromarray(gambar_rgb)).unsqueeze(0)
    with torch.inference_mode():
        probabilitas_awal = torch.softmax(_dapatkan_model()(tensor), dim=1)[0]
    # Menambahkan smoothing untuk menghindari keyakinan berlebih
    probabilitas = 0.9 * probabilitas_awal + 0.1 / len(KELAS_KESEGARAN)
    # Tentukan kelas berdasarkan ambang batas
    busuk_prob = probabilitas[2]
    kurang_segar_prob = probabilitas[1]
    segar_prob = probabilitas[0]
    if busuk_prob > 0.5:
        indeks = 2
    elif kurang_segar_prob > 0.5:
        indeks = 1
    else:
        indeks = 0
    # Hitung skor kesegaran ordinal (0‑100)
    nilai_kelas = torch.tensor((100.0, 50.0, 0.0))
    indeks_kesegaran = float(torch.sum(probabilitas * nilai_kelas))
    return (
        KELAS_KESEGARAN[indeks],
        float(probabilitas[indeks]),
        indeks_kesegaran,
    )
    """Kembalikan kelas, keyakinan kelas, dan indeks kesegaran 0–100."""
    if gambar_bgr is None or gambar_bgr.size == 0:
        raise ValueError("Potongan buah kosong dan tidak bisa diklasifikasikan.")
    gambar_rgb = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2RGB)
    tensor = _transformasi(Image.fromarray(gambar_rgb)).unsqueeze(0)
    with torch.inference_mode():
        probabilitas_awal = torch.softmax(_dapatkan_model()(tensor), dim=1)[0]
    # FruQ-DB tidak memiliki label jenis buah dan foto kondisi di dalamnya
    # beragam. Jangan tampilkan prediksi CNN yang belum dikalibrasi seolah
    # kepastiannya mutlak. Shrink ke distribusi seragam membatasi maksimum
    # keyakinan sekitar 93% dan memberi skor kesegaran yang tidak ekstrem.
    probabilitas = 0.9 * probabilitas_awal + 0.1 / len(KELAS_KESEGARAN)
    indeks = int(torch.argmax(probabilitas))
    # Nilai ordinal mengikuti urutan dataset: busuk=0, kurang segar=50,
    # segar=100. Ekspektasi berbobot menjaga skor rendah untuk busuk walaupun
    # keyakinan model pada kelas busuk sangat tinggi.
    nilai_kelas = torch.tensor((100.0, 50.0, 0.0))
    indeks_kesegaran = float(torch.sum(probabilitas * nilai_kelas))
    return (
        KELAS_KESEGARAN[indeks],
        float(probabilitas[indeks]),
        indeks_kesegaran,
    )
