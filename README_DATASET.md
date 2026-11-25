# Sistem Dataset Buah

## Struktur dan jumlah kelas

Daftar pada `src/deteksi_buah.py` berisi **30 entri**, walaupun permintaan menyebut 29. `cantaloup/cantaloupe` adalah satu kelas alias, bukan dua kelas. Folder menggunakan ID yang aman untuk nama berkas: `cantaloupe`, `coconut`, `date`, `fig`, `kiwi_fruit`, dan seterusnya. Struktur sumbernya:

```text
dataset/<fruit_id>/{segar,kurang_segar,busuk}/
```

Dataset kesegaran FruQ-DB berada di `dataset/segar/`, `dataset/kurang_segar/`, dan `dataset/busuk/` di root dataset. Label sumber Fresh/Mild/Rotten dipetakan ke tiga folder tersebut. Foto-foto ini bercampur jenis buah dan tidak memiliki label jenis per gambar; sumber serta lisensi CC BY 4.0 dicatat di [dataset/FruQ-DB_SOURCE.md](dataset/FruQ-DB_SOURCE.md). Gambar lama seperti `dataset/pisang1.png` tetap tidak digunakan otomatis karena labelnya tidak diketahui.

## Deteksi kesegaran per buah

**Batas penting:** FruQ-DB tidak mencatat jenis buah per foto. Jadi data itu tidak bisa membuktikan bahwa pisang atau buah tertentu busuk; model hanya belajar kondisi umum dari campuran foto buah. Folder `dataset/<fruit_id>/belum_berlabel_kesegaran/` juga tidak dipakai sebagai data segar karena foto Fruits-360 hanya berlabel jenis buah. Untuk model kesegaran per jenis, tambahkan foto tiap buah ke folder kondisi yang benar setelah diperiksa manusia, atau gunakan dataset berlisensi yang memang memberi kedua label (jenis dan kondisi).

Pelatihan menyeimbangkan sampel tiap kelas dan memisahkan validasi secara terstratifikasi. Probabilitas prediksi juga diratakan sedikit supaya model yang belum dikalibrasi tidak melaporkan keyakinan 100%. Bercak cokelat pada kulit pisang sendiri belum membuktikan buah busuk; kematangan dan pembusukan perlu dibedakan lewat contoh berlabel yang sesuai.

`uji_deteksi.py` mendeteksi jenis buah, memotong tiap kotak deteksi, lalu mengklasifikasikan potongan itu sebagai `segar`, `kurang_segar`, atau `busuk`. Hasil GUI ditampilkan sebagai daftar satu baris per buah. Indeks kesegaran 0–100 dihitung sebagai ekspektasi probabilitas kelas dengan nilai busuk=0, kurang segar=50, dan segar=100. Probabilitasnya diratakan sedikit karena keyakinan mentah model belum dikalibrasi; angka itu tetap estimasi, bukan pengukuran fisik kualitas buah. Bobot CNN ada di `model/kesegaran_buah.pt`; latih ulang dari dataset di atas dengan:

```powershell
.venv\Scripts\python.exe scripts\train_freshness.py
```

Classifier dilatih pada foto FruQ-DB yang sudah dipetakan. Validasi yang dicetak skrip memakai pemisahan acak per gambar, bukan pemisahan per video/sesi, sehingga skor validasi dapat lebih tinggi daripada kinerja pada foto baru.

## Mengambil foto jenis buah

Untuk mengunduh hingga 100 foto tiap jenis dari Fruits-360, jalankan:

```powershell
python scripts/unduh_fruits360.py
```

Skrip mengambil hingga 100 foto dari direktori jenis buah yang cocok di [repositori resminya](https://github.com/fruits-360/fruits-360-100x100) melalui API GitHub, bukan mengunduh arsip penuh. Foto disimpan ke `dataset/<fruit_id>/belum_berlabel_kesegaran/`. Sumber menyatakan lisensi CC BY-SA 4.0; pertahankan atribusi dan ketentuan ShareAlike saat membagikan hasil turunan. Manifest `dataset/sumber_fruits360.csv` mencatat file, kelas sumber, split asal, URL, dan lisensi.

Atribusi dataset sumber: Mihai Oltean, *Fruits-360 dataset*, CC BY-SA 4.0. Cantumkan sumber dan lisensi ini saat memublikasikan atau membagikan dataset turunan.

Foto tersebut hanya punya kelas jenis buah, bukan label segar/kurang segar/busuk. Maka foto tetap dipisahkan di folder tanpa label kesegaran dan tidak dihitung oleh laporan kategori kesegaran. Foto Fruits-360 berukuran 100×100, umumnya berlatar putih dan berasal dari objek yang direkam dalam kondisi terkontrol; foto itu bukan pengganti dataset foto kondisi nyata atau anotasi YOLO. Sumber tidak memiliki kelas prune yang terverifikasi, jadi skrip sengaja melewatinya.

Dataset kesegaran legal yang ditemukan seperti FruQ-DB hanya mencakup sebagian jenis buah. Dataset jenis saja tidak digunakan untuk menebak kesegaran.

## Alur kerja

Jalankan dari root project:

```powershell
.venv\Scripts\python.exe scripts/buat_folder_dataset.py
```

Unduh dataset publik secara manual dari sumber yang izinnya telah diperiksa. Catat asal-usul dan label terverifikasi di `dataset_manifest.csv` (contoh header dibuat oleh skrip impor):

```csv
source_path,fruit,condition,source_url,license,attribution
```

`fruit` memakai ID pada `data.yaml`; `condition` harus `segar`, `kurang_segar`, atau `busuk`. `source_path` boleh relatif terhadap root project atau absolut. Kategori kesegaran perlu berasal dari label dataset yang memang sesuai atau tinjauan manusia; label jenis buah saja tidak cukup. Skrip tidak mengambil gambar dari internet, tidak menciptakan label, dan melewati baris tanpa URL sumber/lisensi.

Kemudian jalankan:

```powershell
.venv\Scripts\python.exe scripts/unduh_fruits360.py
.venv\Scripts\python.exe scripts/download_dataset.py
.venv\Scripts\python.exe scripts/hapus_duplikat.py
.venv\Scripts\python.exe scripts/validasi_dataset.py
.venv\Scripts\python.exe scripts/split_dataset.py
.venv\Scripts\python.exe scripts/laporan_dataset.py
```

`hapus_duplikat.py` menghapus berkas dengan SHA-256 identik di seluruh kelas. Validasi membutuhkan OpenCV (`opencv-python`, sudah dipakai project) dan mencatat gambar rusak pada `gambar_rusak.txt`. Split disusun di `dataset_split/{train,val,test}/<fruit_id>/<condition>/`, dengan rasio target 70/20/10 per kombinasi buah/kondisi; pembulatan untuk kelompok kecil dapat mengubah rasio aktual. Data gambar berbeda yang berasal dari objek/sesi foto sama perlu dikelompokkan manual agar tidak bocor antar split. Jalankan split ke direktori output yang bersih agar file lama tidak tertinggal.

Laporan aktual dibuat sebagai `laporan_dataset.csv` dan `laporan_dataset.md`. Target 100 gambar per kategori bukan data yang tersedia saat ini; jangan menyalin atau menggandakan gambar demi memenuhi target.

## YOLO object detection

`data.yaml` memuat **30 kelas** dengan indeks 0–29 mengikuti semua entri pemetaan dalam project. Dataset klasifikasi bertingkat kesegaran di atas bukan format YOLO detection. Untuk melatih deteksi, siapkan terpisah:

```text
dataset_yolo/images/{train,val,test}/
dataset_yolo/labels/{train,val,test}/
```

Setiap gambar memerlukan berkas label `.txt` yang berisi satu baris per objek: `class_id x_center y_center width height`, dengan koordinat ternormalisasi 0–1. Gunakan alat anotasi dan tinjauan manusia; pipeline ini tidak menghasilkan bounding box otomatis. Ubah `path` di `data.yaml` bila dataset YOLO disimpan di lokasi lain. `data.yaml` memakai nama ID stabil; sesuaikan `src/deteksi_buah.py` jika model terlatih memakai nama kelas berbeda.

## Batasan dataset yang tersedia

FruQ-DB memberi label kondisi kesegaran, tetapi label jenis buah per gambar dan kotak anotasi tidak tersedia. Karena itu dataset tersebut melatih classifier kondisi saja; detektor jenis buah tetap memakai model objek terpisah dan tidak dilatih ulang dari gambar FruQ-DB.

## Kandidat dataset buah cacat/berbintik

Dataset yang paling sesuai untuk menambahkan label kualitas terpisah dari `busuk` adalah [Banana and Guava dataset for machine learning and deep learning-based quality classification](https://data.mendeley.com/datasets/56td5w4wz2/2), oleh Abiban Kumari dan Jaswinder Singh (2024), lisensi CC BY 4.0. Dataset menyediakan 1.738 foto asli: pisang (Class A 509, Class B 281, Defect 403) dan jambu biji (Class A 161, Class B 179, Defect 205). Labelnya berdasarkan perubahan fisiologis dan tingkat kualitas; kategori `Defect` bukan anotasi area bercak satu per satu, dan bukan jaminan bahwa semua bercak ringan dibedakan dari busuk. Lihat [artikel dan deskripsi anotasi](https://pmc.ncbi.nlm.nih.gov/articles/PMC11547028/) sebelum pemetaan label.

Untuk apel dan mangga, [Healthy-Defective-Fruits](https://github.com/luischuquim/Healthy-Defective-Fruits) melaporkan masing-masing 5.000 gambar segar dan 5.000 gambar cacat (campuran gambar nyata dan sintetis). Namun halaman repositori metadata Mendeley menyatakan file dataset tidak tersedia, jadi periksa akses dan izin file sebelum mengimpor.

Rekomendasi pemetaan: pertahankan label `segar`, `kurang_segar`, dan `busuk` dari sumber yang memang mendefinisikan kondisi tersebut; tambahkan `bercacat` hanya untuk foto yang sumbernya secara eksplisit memberi label defect. Jangan mengubah semua foto berbercak menjadi `busuk`, dan jangan menganggap label Defect dari banana/guava sebagai label yang berlaku untuk seluruh jenis buah.
