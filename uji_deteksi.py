import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2

from src.deteksi_buah import deteksi_buah

# ====== Palet Golden Dark ======
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
    hex_warna = hex_warna.lstrip("#")
    return tuple(int(hex_warna[i:i + 2], 16) for i in (0, 2, 4))


def _campur(c1, c2, t):
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
        self.root = root
        self.root.title("FruitVision | Deteksi Buah")
        self.root.geometry("1100x760")
        self.root.minsize(900, 620)
        self.root.configure(bg=BG)

        self.lokasi_foto = None
        self.foto_tampil = None
        self._gambar_asli = None
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
            anchor="w"
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
        lokasi = filedialog.askopenfilename(
            title="Pilih Foto Buah",
            filetypes=[
                ("File gambar", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("Semua file", "*.*")
            ]
        )

        if not lokasi:
            return

        # Simpan lokasi foto yang dipilih
        self.lokasi_foto = lokasi

        # Tampilkan foto sebelum deteksi
        try:
            with Image.open(lokasi) as foto:
                self._gambar_asli = foto.convert("RGB")
            self._perbarui_pratinjau()
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

    def proses_deteksi(self):
        if not self.lokasi_foto:
            messagebox.showwarning(
                "Foto Belum Dipilih",
                "Silakan pilih foto terlebih dahulu."
            )
            return

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
        threading.Thread(
            target=self._jalankan_deteksi,
            args=(self.lokasi_foto,),
            daemon=True
        ).start()

    def _jalankan_deteksi(self, lokasi_foto):
        try:
            hasil = deteksi_buah(lokasi_foto)
        except Exception as error:
            self._hasil_deteksi.put(("error", error))
        else:
            self._hasil_deteksi.put(("sukses", hasil))

    def _periksa_hasil_deteksi(self):
        try:
            status, hasil = self._hasil_deteksi.get_nowait()
        except queue.Empty:
            pass
        else:
            self.tombol_pilih_foto.config(state=tk.NORMAL)
            self.tombol_deteksi.config(state=tk.NORMAL)

            if status == "error":
                self.label_ringkasan.config(text="Deteksi gagal")
                messagebox.showerror("Deteksi Gagal", str(hasil))
            else:
                try:
                    self._tampilkan_hasil_deteksi(*hasil)
                except Exception as error:
                    messagebox.showerror("Deteksi Gagal", str(error))
        finally:
            self.root.after(100, self._periksa_hasil_deteksi)

    def _tampilkan_hasil_deteksi(self, gambar, daftar_buah):
        ringkasan = {}
        for buah in daftar_buah:
            data = ringkasan.setdefault(
                buah["nama"], {"jumlah": 0, "keyakinan": []}
            )
            data["jumlah"] += 1
            data["keyakinan"].append(buah["keyakinan_deteksi"])

        jumlah_jenis = len(ringkasan)
        jumlah_buah = len(daftar_buah)
        self.label_ringkasan.config(
            text=f"{jumlah_buah} buah  •  {jumlah_jenis} jenis"
        )
        self._gambar_asli = Image.fromarray(
            cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)
        )
        self._perbarui_pratinjau()

        if not ringkasan:
            self._isi_daftar_hasil("Buah Tidak Terdeteksi.")
            return

        baris = [
            f"{'JENIS':<17} {'QTY':>3}  {'CONF':>4}",
            "-" * 28
        ]
        urutan = sorted(
            ringkasan.items(),
            key=lambda item: (-item[1]["jumlah"], item[0].lower())
        )
        for nama, data in urutan:
            jumlah = data["jumlah"]
            keyakinan_tertinggi = max(data["keyakinan"])
            baris.append(
                f"{nama[:17]:<17} {jumlah:>3}  {keyakinan_tertinggi:>3.0%}"
            )

        self._isi_daftar_hasil("\n".join(baris), heading=True)

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
    root = tk.Tk()
    aplikasi = AplikasiDeteksiBuah(root)
    root.mainloop()