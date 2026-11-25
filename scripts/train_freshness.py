"""Latih classifier kesegaran dari dataset/Fresh-Mild-Rotten yang sudah dipetakan."""

import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFile
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.kesegaran import (
    CNNKesegaran, KELAS_KESEGARAN, MODEL_KESEGARAN_PATH, UKURAN_INPUT,
)


DATASET = ROOT / "dataset"
EKSTENSI_GAMBAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ImageFile.LOAD_TRUNCATED_IMAGES = False


class DatasetKesegaran(Dataset):
    def __init__(self, berkas, transformasi):
        self.berkas = berkas
        self.transformasi = transformasi

    def __len__(self):
        return len(self.berkas)

    def __getitem__(self, indeks):
        path, label = self.berkas[indeks]
        with Image.open(path) as gambar:
            gambar = gambar.convert("RGB")
        return self.transformasi(gambar), label


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.set_num_interop_threads(1)
    berkas = []
    for label, kelas in enumerate(KELAS_KESEGARAN):
        folder = DATASET / kelas
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {folder}")
        gambar_kelas = [
            (path, label) for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in EKSTENSI_GAMBAR
        ]
        if not gambar_kelas:
            raise ValueError(f"Tidak ada gambar di folder {folder}")
        berkas.extend(gambar_kelas)
        print(f"{kelas}: {len(gambar_kelas)} gambar")

    acak = transforms.Compose((
        transforms.Resize((UKURAN_INPUT, UKURAN_INPUT)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.12),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ))
    biasa = transforms.Compose((
        transforms.Resize((UKURAN_INPUT, UKURAN_INPUT)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ))
    # Pisahkan setiap kelas secara terstratifikasi supaya kelas terkecil tetap
    # terwakili di validasi dan evaluasi tidak didominasi kelas terbesar.
    generator = torch.Generator().manual_seed(42)
    indeks_latih, indeks_validasi = [], []
    for label in range(len(KELAS_KESEGARAN)):
        anggota_kelas = [i for i, (_, y) in enumerate(berkas) if y == label]
        urutan = torch.randperm(len(anggota_kelas), generator=generator).tolist()
        indeks_kelas = [anggota_kelas[j] for j in urutan]
        batas = max(1, int(len(indeks_kelas) * 0.85))
        indeks_latih.extend(indeks_kelas[:batas])
        indeks_validasi.extend(indeks_kelas[batas:])
    latih = torch.utils.data.Subset(
        DatasetKesegaran(berkas, acak), indeks_latih
    )
    validasi = torch.utils.data.Subset(
        DatasetKesegaran(berkas, biasa), indeks_validasi
    )
    frekuensi_latih = torch.bincount(torch.tensor([
        berkas[i][1] for i in indeks_latih
    ]), minlength=len(KELAS_KESEGARAN)).float()
    bobot_sampel = torch.tensor([
        1.0 / frekuensi_latih[berkas[i][1]] for i in indeks_latih
    ])
    sampler = WeightedRandomSampler(
        bobot_sampel, num_samples=len(indeks_latih), replacement=True,
        generator=generator,
    )
    loader_latih = DataLoader(
        latih, batch_size=48, sampler=sampler, num_workers=0
    )
    loader_validasi = DataLoader(validasi, batch_size=64, shuffle=False, num_workers=0)

    model = CNNKesegaran()
    optim = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    rugi = nn.CrossEntropyLoss()
    terbaik = 0.0
    MODEL_KESEGARAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 9):
        model.train()
        total_rugi = 0.0
        for gambar, label in loader_latih:
            optim.zero_grad(set_to_none=True)
            keluaran = model(gambar)
            loss = rugi(keluaran, label)
            loss.backward()
            optim.step()
            total_rugi += loss.item() * len(label)

        model.eval()
        benar = 0
        total = 0
        with torch.inference_mode():
            for gambar, label in loader_validasi:
                prediksi = model(gambar).argmax(1)
                benar += int((prediksi == label).sum())
                total += len(label)
        akurasi = benar / max(total, 1)
        print(
            f"Epoch {epoch:02d}/08 | "
            f"loss {total_rugi / jumlah_latih:.4f} | "
            f"validasi {akurasi:.1%}",
            flush=True,
        )
        if akurasi > terbaik:
            terbaik = akurasi
            torch.save({
                "state_dict": model.state_dict(),
                "kelas": KELAS_KESEGARAN,
                "ukuran_input": UKURAN_INPUT,
                "akurasi_validasi_acak": terbaik,
                "jumlah_gambar": len(berkas),
                "jumlah_latih_per_kelas": frekuensi_latih.to(torch.int).tolist(),
            }, MODEL_KESEGARAN_PATH)

    print(f"Model terbaik tersimpan: {MODEL_KESEGARAN_PATH} ({terbaik:.1%})")
    print("Validasi ini acak per gambar; gambar berurutan dari sumber video dapat serupa.")


if __name__ == "__main__":
    main()
