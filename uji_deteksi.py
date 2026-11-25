import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np  # BARU

from src.deteksi_buah import deteksi_buah

# ====== Palet Golden Dark ======
# Warna-warna ini dipakai bersama agar tampilan seluruh aplikasi konsisten.
BG = "#0B0905"
KARTU = "#17130B"
PANEL = "#0F0C07"
EMAS = "#F2C94C"
EMAS_TERANG = "#FFE08A"
EMAS_REDUP = "#4A3A12"
BAYANGAN = "#030201"
TEKS = "#F3E3B0"
TEKS_REDUP = "#A89868"


def _rgb(hex_warna):
    # Ubah warna format #RRGGBB menjadi komponen merah, hijau, dan biru.
    hex_warna = hex_warna.lstrip("#")
    return tuple(int(hex_warna[i:i + 2], 16) for i in (0, 2, 4))


def _campur(c1, c2, t):
    # Campur dua warna dengan proporsi t untuk membuat gradasi tombol.
    a, b = _rgb(c1), _rgb(c2)
    return "#%02x%02x%02x" % tuple(
        int(a[i] + (b[i] - a[i]) * t) for i in range(3)
    )


class Tombol3D(tk.Canvas):
    """Tombol dengan gradasi, sisi tebal, dan efek tekan ala 3D."""

    TEBAL = 6

    def __init__(self, parent, text, command, atas, bawah, sisi, fg,
                 lebar=150, tinggi=46, bg=BG):
        super().__init__(
            parent, width=lebar, height=tinggi, bg=bg,
            highlightthickness=0, bd=0, cursor="hand2"
        )
        self._text = text
        self._command = command
        self._atas, self._bawah, self._sisi, self._fg = atas, bawah, sisi, fg
        self._lebar, self._tinggi = lebar, tinggi
        self._tekan = False
        self._hover = False
        self._aktif = True

        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, tekan=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(tekan=True))
        self.bind("<ButtonRelease-1>", self._lepas)
        self._gambar()

    def _set(self, hover=None, tekan=None):
        if hover is not None:
            self._hover = hover
        if tekan is not None:
            self._tekan = tekan
        self._gambar()

    def _lepas(self, event):
        if not (self._tekan and self._aktif):
            return
        self._set(tekan=False)
        if (
            0 <= event.x <= self._lebar
            and 0 <= event.y <= self._tinggi
        ):
            self._command()

    def configure(self, cnf=None, **kw):
        if "state" in kw:
            self._aktif = kw.pop("state") != tk.DISABLED
            super().configure(cursor="hand2" if self._aktif else "arrow")
            self._gambar()
        if cnf or kw:
            super().configure(cnf, **kw)

    config = configure

    def _gambar(self):
        # Gambar ulang tombol sesuai status aktif, hover, dan sedang ditekan.
        self.delete("all")
        w, h, t = self._lebar, self._tinggi, self.TEBAL
        bh = h - t
        atas, bawah, sisi, fg = self._atas, self._bawah, self._sisi, self._fg

        if not self._aktif:
            atas, bawah, sisi, fg = "#3A3320", "#2A2515", "#12100A", "#7A7050"
        elif self._hover and not self._tekan:
            atas = _campur(atas, "#FFFFFF", 0.18)
            bawah = _campur(bawah, "#FFFFFF", 0.12)

        off = t - 2 if self._tekan else 0

        # bayangan jatuh
        self.create_rectangle(
            3, 5, w, h, fill=BAYANGAN, outline=BAYANGAN
        )
        # sisi tebal (ketebalan tombol)
        self.create_rectangle(
            0, bh // 2 + 2, w - 3, h - 3, fill=sisi, outline=sisi
        )
        # badan bergradasi
        for i in range(bh):
            self.create_line(
                0, off + i, w - 3, off + i,
                fill=_campur(atas, bawah, i / max(bh - 1, 1))
            )
        # kilau tepi atas & kiri
        self.create_line(
            0, off, w - 3, off, fill=_campur(atas, "#FFFFFF", 0.55)
        )
        self.create_line(
            0, off, 0, off + bh, fill=_campur(atas, "#FFFFFF", 0.3)
        )
        # tepi bawah gelap
        self.create_line(
            0, off + bh - 1, w - 3, off + bh - 1,
            fill=_campur(bawah, "#000000", 0.35)
        )
        # teks dengan bayangan kecil
        cx, cy = (w - 3) / 2, off + bh / 2
        font = ("Segoe UI", 10, "bold")
        self.create_text(
            cx + 1, cy + 1, text=self._text, font=font,
            fill=_campur(atas, "#000000", 0.5)
        )
        self.create_text(cx, cy, text=self._text, font=font, fill=fg)


class AplikasiDeteksiBuah:
    def __init__(self, root):
        # Siapkan jendela, status aplikasi, lalu susun komponen antarmuka.
        self.root = root
        self.root.title("FruitVision | Deteksi Buah")
        self.root.geometry("1100x760")
        self.root.minsize(900, 620)
        self.root.configure(bg=BG)

        self.lokasi_foto = None
        self.foto_tampil = None
        self._gambar_asli = None
        self._gambar_hasil_deteksi = None
        self._gambar_sumber = None  # BARU
        self._info_citra = None  # BARU
        self._hasil_deteksi = queue.Queue()
        self.root.after(100, self._periksa_hasil_deteksi)

        konten = tk.Frame(root, bg=BG)
        konten.pack(fill="both", expand=True, padx=26, pady=22)

        # ---- Kepala dengan judul timbul 3D ----
        kepala = tk.Frame(konten, bg=BG)
        kepala.pack(fill="x", pady=(0, 16))
        judul = tk.Canvas(
            kepala, width=360, height=52, bg=BG, highlightthickness=0
        )
        judul.pack(anchor="w")
        font_judul = ("Segoe UI", 27, "bold")
        for dx, dy, warna in (
            (5, 5, "#000000"), (4, 4, "#2A1F08"), (3, 3, "#5A4210"),
            (2, 2, "#8A6514"), (1, 1, "#B8860B")
        ):
            judul.create_text(
                2 + dx, 26 + dy, text="FruitVision",
                font=font_judul, fill=warna, anchor="w"
            )
        judul.create_text(
            2, 26, text="FruitVision", font=font_judul,
            fill=EMAS_TERANG, anchor="w"
        )
        tk.Label(
            kepala,
            text="Deteksi buah dengan hasil yang jelas dan mudah dibaca",
            font=("Segoe UI", 10),
            bg=BG,
            fg=TEKS_REDUP
        ).pack(anchor="w", pady=(2, 8))
        tk.Frame(kepala, bg=EMAS, height=1).pack(fill="x")
        tk.Frame(kepala, bg="#000000", height=2).pack(fill="x")

        # ---- Kontrol ----
        kontrol = tk.Frame(konten, bg=BG)
        kontrol.pack(fill="x", pady=(0, 14))
        self.tombol_pilih_foto = Tombol3D(
            kontrol, "Pilih Foto", self.pilih_foto,
            atas="#F7D774", bawah="#C9961A", sisi="#7A5A0B", fg="#1A1304"
        )
        self.tombol_pilih_foto.pack(side="left")

        self.tombol_deteksi = Tombol3D(
            kontrol, "Deteksi Buah", self.proses_deteksi,
            atas="#B8801F", bawah="#7F5512", sisi="#3F2A09", fg="#FFF3CC"
        )
        self.tombol_deteksi.pack(side="left", padx=(14, 0))

        # ---- Isi ----
        isi = tk.Frame(konten, bg=BG)
        isi.pack(fill="both", expand=True)
        isi.grid_columnconfigure(0, weight=3, minsize=460)
        isi.grid_columnconfigure(1, weight=2, minsize=300)
        isi.grid_rowconfigure(0, weight=1)

        bungkus_gambar, kartu_gambar = self._kartu_3d(isi)
        bungkus_gambar.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        tk.Label(
            kartu_gambar,
            text="PRATINJAU GAMBAR",
            font=("Segoe UI", 9, "bold"),
            bg=KARTU,
            fg=EMAS
        ).pack(anchor="w", padx=16, pady=(12, 8))

        self.label_gambar = tk.Label(
            kartu_gambar,
            text="Pilih foto buah untuk memulai",
            font=("Segoe UI", 11),
            bg=PANEL,
            fg=TEKS_REDUP,
            relief="sunken",
            bd=3
        )
        self.label_gambar.pack(
            fill="both", expand=True, padx=12, pady=(0, 14)
        )
        self.label_gambar.bind(
            "<Configure>", self._perbarui_pratinjau
        )

        bungkus_hasil, kartu_hasil = self._kartu_3d(isi)
        bungkus_hasil.grid(row=0, column=1, sticky="nsew", padx=(10, 0))
        tk.Label(
            kartu_hasil,
            text="HASIL DETEKSI",
            font=("Segoe UI", 9, "bold"),
            bg=KARTU,
            fg=EMAS
        ).pack(anchor="w", padx=16, pady=(12, 2))
        self.label_ringkasan = tk.Label(
            kartu_hasil,
            text="Pilih foto untuk melihat hasil",
            font=("Segoe UI", 10),
            bg=KARTU,
            fg=TEKS,
            anchor="w",
            justify="left",
            wraplength=270
        )
        self.label_ringkasan.pack(fill="x", padx=16, pady=(0, 10))

        daftar_frame = tk.Frame(kartu_hasil, bg=KARTU)
        daftar_frame.pack(
            fill="both", expand=True, padx=(16, 10), pady=(0, 16)
        )
        scrollbar = tk.Scrollbar(
            daftar_frame, orient="vertical",
            bg="#2A2110", troughcolor=PANEL, activebackground=EMAS,
            relief="raised", bd=1
        )
        scrollbar.pack(side="right", fill="y")
        self.daftar_hasil = tk.Text(
            daftar_frame,
            height=10,
            wrap="none",
            state="disabled",
            font=("Consolas", 9),
            bg=PANEL,
            fg=TEKS,
            relief="sunken",
            bd=3,
            padx=10,
            pady=10,
            yscrollcommand=scrollbar.set,
            cursor="arrow",
            highlightthickness=0
        )
        self.daftar_hasil.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.daftar_hasil.yview)
        self.daftar_hasil.tag_configure(
            "judul", font=("Consolas", 9, "bold"), foreground=EMAS
        )
        self.daftar_hasil.tag_configure(
            "buah", foreground=TEKS
        )
        self.daftar_hasil.tag_configure(
            "kosong", foreground=TEKS_REDUP
        )
        self.tombol_download = Tombol3D(
            kartu_hasil, "Download Hasil Deteksi", self.download_hasil_deteksi,
            atas="#F7D774", bawah="#C9961A", sisi="#7A5A0B", fg="#1A1304",
            lebar=240, tinggi=44, bg=KARTU
        )
        self.tombol_download.pack(anchor="e", padx=16, pady=(0, 14))
        self.tombol_download.config(state=tk.DISABLED)
        self.tombol_download.bind(
            "<ButtonRelease-1>",
            self._beri_pesan_jika_hasil_belum_tersedia,
            add="+"
        )

        # BARU: panel nilai citra berada di bawah kedua kartu.
        bungkus_nilai, kartu_nilai = self._kartu_3d(konten)
        bungkus_nilai.configure(height=210)
        bungkus_nilai.pack_propagate(False)
        bungkus_nilai.pack(fill="x", pady=(12, 0))
        tk.Label(
            kartu_nilai,
            text="NILAI CITRA",
            font=("Segoe UI", 9, "bold"),
            bg=KARTU,
            fg=EMAS
        ).pack(anchor="w", padx=16, pady=(10, 6))
        nilai_frame = tk.Frame(kartu_nilai, bg=KARTU)
        nilai_frame.pack(
            fill="both", expand=True, padx=(16, 10), pady=(0, 10)
        )
        scrollbar_nilai = tk.Scrollbar(
            nilai_frame, orient="vertical",
            bg="#2A2110", troughcolor=PANEL, activebackground=EMAS,
            relief="raised", bd=1
        )
        scrollbar_nilai.pack(side="right", fill="y")
        self.panel_nilai_citra = tk.Text(
            nilai_frame,
            height=6,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            bg=PANEL,
            fg=TEKS,
            relief="sunken",
            bd=3,
            padx=10,
            pady=8,
            yscrollcommand=scrollbar_nilai.set,
            cursor="arrow",
            highlightthickness=0
        )
        self.panel_nilai_citra.pack(side="left", fill="both", expand=True)
        scrollbar_nilai.config(command=self.panel_nilai_citra.yview)
        self._isi_panel_nilai_citra(
            "Pilih foto, lalu tekan Deteksi Buah untuk menampilkan nilainya."
        )

    def _isi_panel_nilai_citra(self, teks):  # BARU
        self.panel_nilai_citra.config(state="normal")  # BARU
        self.panel_nilai_citra.delete("1.0", tk.END)  # BARU
        self.panel_nilai_citra.insert(tk.END, teks)  # BARU
        self.panel_nilai_citra.config(state="disabled")  # BARU
        self.panel_nilai_citra.yview_moveto(0)  # BARU

    @staticmethod
    def _statistik_rgb_hsv(gambar_rgb):  # BARU
        # Hitung ringkasan warna RGB dan HSV untuk seluruh gambar atau area buah.
        piksel_rgb = np.asarray(gambar_rgb.convert("RGB"), dtype=np.uint8)  # BARU
        piksel_hsv = cv2.cvtColor(piksel_rgb, cv2.COLOR_RGB2HSV)  # BARU
        hsv_mean = piksel_hsv.mean(axis=(0, 1))  # BARU
        hsv_min = piksel_hsv.min(axis=(0, 1)).astype(np.float64)  # BARU
        hsv_max = piksel_hsv.max(axis=(0, 1)).astype(np.float64)  # BARU
        hsv_std = piksel_hsv.std(axis=(0, 1))  # BARU
        jumlah_hue = np.bincount(  # BARU: hitung hue rata-rata secara melingkar.
            piksel_hsv[..., 0].ravel(), minlength=180
        )
        sudut_h = np.deg2rad(np.arange(180) * 2.0)  # BARU
        hue_rata = np.degrees(np.arctan2(  # BARU
            (np.sin(sudut_h) * jumlah_hue).sum(),
            (np.cos(sudut_h) * jumlah_hue).sum()
        )) % 360.0
        hsv_mean[0] = hue_rata  # BARU: H dalam derajat (0-360).
        hsv_min[0] *= 2.0  # BARU
        hsv_max[0] *= 2.0  # BARU
        hsv_std[0] *= 2.0  # BARU
        hsv_mean[1:] *= 100.0 / 255.0  # BARU: S dan V dalam persen.
        hsv_min[1:] *= 100.0 / 255.0  # BARU
        hsv_max[1:] *= 100.0 / 255.0  # BARU
        hsv_std[1:] *= 100.0 / 255.0  # BARU
        return {  # BARU
            "rgb_mean": piksel_rgb.mean(axis=(0, 1)),  # BARU
            "rgb_min": piksel_rgb.min(axis=(0, 1)),  # BARU
            "rgb_max": piksel_rgb.max(axis=(0, 1)),  # BARU
            "rgb_std": piksel_rgb.std(axis=(0, 1)),  # BARU
            "hsv_mean": hsv_mean,  # BARU
            "hsv_min": hsv_min,  # BARU
            "hsv_max": hsv_max,  # BARU
            "hsv_std": hsv_std  # BARU
        }  # BARU

    def _tampilkan_nilai_citra(self, gambar_rgb, daftar_buah):  # BARU
        # Susun statistik gambar dan setiap kotak deteksi untuk panel nilai citra.
        try:  # BARU
            statistik = self._statistik_rgb_hsv(gambar_rgb)  # BARU
            lebar, tinggi = gambar_rgb.size  # BARU
            mode = (self._info_citra or (lebar, tinggi, gambar_rgb.mode))[2]  # BARU
            nama_rgb = ("R", "G", "B")  # BARU
            nama_hsv = ("H", "S", "V")  # BARU
            baris = [  # BARU
                "INFO CITRA",
                f"Ukuran        : {lebar} x {tinggi} piksel",
                f"Jumlah piksel : {lebar * tinggi:,}".replace(",", "."),
                f"Mode warna    : {mode}",
                "",
                "RATA-RATA SELURUH GAMBAR (RGB 0-255; HSV H derajat, S/V persen)",
                "RGB           : " + "  ".join(
                    f"{nama}={nilai:.2f}"
                    for nama, nilai in zip(nama_rgb, statistik["rgb_mean"])
                ),
                "HSV           : " + "  ".join(
                    f"{nama}={nilai:.2f}{' deg' if nama == 'H' else '%'}"
                    for nama, nilai in zip(nama_hsv, statistik["hsv_mean"])
                ),
                "",
                "MIN / MAX / STANDAR DEVIASI PER SALURAN",
            ]  # BARU
            for ruang, nama, satuan in (  # BARU
                ("RGB", nama_rgb, ""),
                ("HSV", nama_hsv, " deg"),
            ):
                min_key = f"{ruang.lower()}_min"  # BARU
                max_key = f"{ruang.lower()}_max"  # BARU
                std_key = f"{ruang.lower()}_std"  # BARU
                for indeks, saluran in enumerate(nama):  # BARU
                    unit = satuan if saluran == "H" else ("%" if ruang == "HSV" else "")  # BARU
                    baris.append(  # BARU
                        f"{ruang} {saluran}: min={statistik[min_key][indeks]:.2f}{unit}, "
                        f"max={statistik[max_key][indeks]:.2f}{unit}, "
                        f"std={statistik[std_key][indeks]:.2f}{unit}"
                    )  # BARU
            baris.extend(("", "NILAI PER BUAH (AREA KOTAK DETEKSI)"))  # BARU
            for nomor, buah in enumerate(daftar_buah, start=1):  # BARU
                x1, y1, x2, y2 = map(int, buah["koordinat"])  # BARU
                x1 = max(0, min(lebar, x1))  # BARU
                x2 = max(0, min(lebar, x2))  # BARU
                y1 = max(0, min(tinggi, y1))  # BARU
                y2 = max(0, min(tinggi, y2))  # BARU
                baris.append(f"{nomor}. {buah['nama']}")  # BARU
                if x2 <= x1 or y2 <= y1:  # BARU
                    baris.append("   Area kotak tidak valid; statistik tidak tersedia.")  # BARU
                    continue  # BARU
                area = gambar_rgb.crop((x1, y1, x2, y2))  # BARU
                statistik_area = self._statistik_rgb_hsv(area)  # BARU
                baris.append(  # BARU
                    "   RGB: " + "  ".join(
                        f"{nama}={nilai:.2f}"
                        for nama, nilai in zip(nama_rgb, statistik_area["rgb_mean"])
                    )
                )  # BARU
                baris.append(  # BARU
                    "   HSV: " + "  ".join(
                        f"{nama}={nilai:.2f}{' deg' if nama == 'H' else '%'}"
                        for nama, nilai in zip(nama_hsv, statistik_area["hsv_mean"])
                    )
                )  # BARU
            self._isi_panel_nilai_citra("\n".join(baris))  # BARU
        except Exception as error:  # BARU
            self._isi_panel_nilai_citra(  # BARU
                f"Nilai citra gagal dihitung: {error}"
            )  # BARU

    def _kartu_3d(self, parent, kedalaman=8):
        """Kartu timbul: bayangan bergeser + kilau emas di tepi atas."""
        bungkus = tk.Frame(parent, bg=BG)
        bayangan = tk.Frame(bungkus, bg=BAYANGAN)
        bayangan.place(
            x=kedalaman, y=kedalaman, relwidth=1, relheight=1,
            width=-kedalaman, height=-kedalaman
        )
        kartu = tk.Frame(
            bungkus, bg=KARTU, highlightthickness=1,
            highlightbackground=EMAS_REDUP
        )
        kartu.place(
            x=0, y=0, relwidth=1, relheight=1,
            width=-kedalaman, height=-kedalaman
        )
        tk.Frame(kartu, bg=EMAS, height=2).pack(fill="x")
        return bungkus, kartu

    def pilih_foto(self):
        # Minta pengguna memilih gambar, simpan salinannya, lalu tampilkan pratinjau.
        lokasi = filedialog.askopenfilename(
            title="Pilih Foto Buah",
            filetypes=[
                ("File gambar", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Semua file", "*.*")
            ]
        )

        if not lokasi:
            return

        self._gambar_hasil_deteksi = None
        self.tombol_download.config(state=tk.DISABLED)

        # Simpan lokasi foto yang dipilih
        self.lokasi_foto = lokasi

        # Tampilkan foto sebelum deteksi
        try:
            with Image.open(lokasi) as foto:
                self._info_citra = (foto.width, foto.height, foto.mode)  # BARU
                self._gambar_sumber = foto.convert("RGB")  # BARU
                self._gambar_asli = self._gambar_sumber.copy()
            self._perbarui_pratinjau()
            self._isi_panel_nilai_citra(  # BARU
                "Foto diperbarui. Nilai citra akan muncul setelah deteksi selesai."
            )
            self.label_ringkasan.config(
                text="Foto siap dianalisis"
            )
            self._isi_daftar_hasil(
                "Tekan tombol Deteksi buah untuk mulai."
            )
        except Exception as error:
            messagebox.showerror(
                "Kesalahan Foto",
                f"Foto tidak dapat dibuka:\n{error}"
            )
            self.lokasi_foto = None
            self._gambar_asli = None
            self._gambar_hasil_deteksi = None
            self._gambar_sumber = None  # BARU
            self._info_citra = None  # BARU
            self._isi_panel_nilai_citra("Pilih foto yang dapat dibuka.")  # BARU

    def proses_deteksi(self):
        # Validasi pilihan foto dan jalankan deteksi di thread agar UI tetap responsif.
        if not self.lokasi_foto:
            messagebox.showwarning(
                "Foto Belum Dipilih",
                "Silakan pilih foto terlebih dahulu."
            )
            return

        self._gambar_hasil_deteksi = None
        self.tombol_download.config(state=tk.DISABLED)
        self.tombol_pilih_foto.config(state=tk.DISABLED)
        self.tombol_deteksi.config(state=tk.DISABLED)
        self.label_ringkasan.config(
            text=(
                "Menyiapkan model dan memeriksa gambar..."
            )
        )
        self._isi_daftar_hasil(
            "Pertama kali, model sekitar 50 MB diunduh satu kali."
        )
        self._isi_panel_nilai_citra("Deteksi berjalan; nilai citra akan dihitung...")  # BARU
        threading.Thread(
            target=self._jalankan_deteksi,
            args=(self.lokasi_foto,),
            daemon=True
        ).start()

    def _jalankan_deteksi(self, lokasi_foto):
        # Jalankan model di thread latar dan kirim hasilnya melalui antrean.
        try:
            hasil = deteksi_buah(lokasi_foto)
        except Exception as error:
            self._hasil_deteksi.put(("error", error))
        else:
            self._hasil_deteksi.put(("sukses", hasil))

    def _periksa_hasil_deteksi(self):
        # Ambil hasil thread secara berkala dan perbarui komponen UI di thread utama.
        try:
            status, hasil = self._hasil_deteksi.get_nowait()
        except queue.Empty:
            pass
        else:
            self.tombol_pilih_foto.config(state=tk.NORMAL)
            self.tombol_deteksi.config(state=tk.NORMAL)

            if status == "error":
                self._gambar_hasil_deteksi = None
                self.tombol_download.config(state=tk.DISABLED)
                self.label_ringkasan.config(text="Deteksi gagal")
                self._isi_panel_nilai_citra(  # BARU
                    "Nilai citra belum tersedia karena deteksi gagal."
                )
                messagebox.showerror("Deteksi Gagal", str(hasil))
            else:
                try:
                    self._tampilkan_hasil_deteksi(*hasil)
                except Exception as error:
                    messagebox.showerror("Deteksi Gagal", str(error))
        finally:
            self.root.after(100, self._periksa_hasil_deteksi)

    def _tampilkan_hasil_deteksi(self, gambar, daftar_buah):
        # Tampilkan gambar beranotasi, ringkasan kondisi, dan nilai warna citra.
        gambar_sumber = (  # BARU: statistik memakai foto asli tanpa anotasi.
            self._gambar_sumber.copy()
            if self._gambar_sumber is not None else None
        )
        jumlah_buah = len(daftar_buah)
        jumlah_kesegaran = {
            kelas: sum(buah["kesegaran"] == kelas for buah in daftar_buah)
            for kelas in ("segar", "kurang_segar", "busuk")
        }
        self.label_ringkasan.config(
            text=(
                f"{jumlah_buah} buah  •  Segar {jumlah_kesegaran['segar']}  "
                f"•  Kurang segar {jumlah_kesegaran['kurang_segar']}  "
                f"•  Busuk {jumlah_kesegaran['busuk']}"
            )
        )
        self._gambar_hasil_deteksi = Image.fromarray(
            cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
        )
        self._gambar_asli = self._gambar_hasil_deteksi.copy()
        self._perbarui_pratinjau()
        if gambar_sumber is not None:  # BARU
            self._tampilkan_nilai_citra(gambar_sumber, daftar_buah)  # BARU

        if not daftar_buah:
            self._isi_daftar_hasil("Buah Tidak Terdeteksi.")
        else:
            baris = ["DAFTAR BUAH (SETIAP BUAH DINILAI TERPISAH)", ""]
            for nomor, buah in enumerate(daftar_buah, start=1):
                baris.append(
                    f"{nomor}. {buah['nama']} — {buah['kesegaran'].replace('_', ' ').title()}"
                )
                baris.append(
                    f"   Indeks kesegaran {buah['indeks_kesegaran']:.0f}%  •  "
                    f"Keyakinan kondisi {buah['keyakinan_kesegaran']:.0%}"
                )
                baris.append("")

            self._isi_daftar_hasil("\n".join(baris).rstrip(), heading=True)

        self.tombol_download.config(state=tk.NORMAL)

    def _beri_pesan_jika_hasil_belum_tersedia(self, _event=None):
        if self._gambar_hasil_deteksi is None:
            self.tombol_download._set(tekan=False)
            self._tampilkan_pesan_hasil_belum_tersedia()

    @staticmethod
    def _tampilkan_pesan_hasil_belum_tersedia():
        messagebox.showinfo(
            "Hasil Belum Tersedia",
            "Gambar hasil deteksi belum tersedia untuk diunduh."
        )

    def download_hasil_deteksi(self):
        # Simpan gambar hasil deteksi dengan format PNG atau JPEG yang didukung.
        if self._gambar_hasil_deteksi is None:
            self._tampilkan_pesan_hasil_belum_tersedia()
            return

        lokasi = filedialog.asksaveasfilename(
            title="Simpan Hasil Deteksi",
            initialfile="FruitVision_Hasil_Deteksi.png",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg")
            ]
        )
        if not lokasi:
            return

        path = Path(lokasi)
        if not path.suffix:
            path = path.with_suffix(".png")
        ekstensi = path.suffix.lower()
        if ekstensi not in {".png", ".jpg", ".jpeg"}:
            messagebox.showerror(
                "Format Tidak Didukung",
                "Simpan gambar dengan format PNG atau JPG."
            )
            return

        try:
            if ekstensi in {".jpg", ".jpeg"}:
                self._gambar_hasil_deteksi.convert("RGB").save(
                    path,
                    format="JPEG",
                    quality=95,
                    subsampling=0,
                    optimize=True
                )
            else:
                self._gambar_hasil_deteksi.save(path, format="PNG")
        except Exception as error:
            messagebox.showerror(
                "Gagal Menyimpan",
                f"Gambar hasil deteksi tidak dapat disimpan:\n{error}"
            )
        else:
            messagebox.showinfo(
                "Berhasil Disimpan",
                f"Gambar hasil deteksi disimpan di:\n{path}"
            )

    def _isi_daftar_hasil(self, teks, heading=False):
        self.daftar_hasil.config(state="normal")
        self.daftar_hasil.delete("1.0", tk.END)
        if heading:
            baris, isi = teks.split("\n", 1)
            self.daftar_hasil.insert(tk.END, baris + "\n", "judul")
            self.daftar_hasil.insert(tk.END, isi, "buah")
        else:
            self.daftar_hasil.insert(tk.END, teks, "kosong")
        self.daftar_hasil.config(state="disabled")
        self.daftar_hasil.yview_moveto(0)

    def _perbarui_pratinjau(self, _event=None):
        # Sesuaikan ukuran pratinjau dengan ruang label tanpa mengubah rasio gambar.
        if self._gambar_asli is None:
            return

        lebar = self.label_gambar.winfo_width() - 24
        tinggi = self.label_gambar.winfo_height() - 24
        if lebar <= 1 or tinggi <= 1:
            return

        foto = self._gambar_asli.copy()
        foto.thumbnail((lebar, tinggi), Image.Resampling.LANCZOS)
        self.foto_tampil = ImageTk.PhotoImage(foto)
        self.label_gambar.config(image=self.foto_tampil, text="")


if __name__ == "__main__":
    # Titik masuk aplikasi saat file dijalankan secara langsung.
    root = tk.Tk()
    aplikasi = AplikasiDeteksiBuah(root)
    root.mainloop()
