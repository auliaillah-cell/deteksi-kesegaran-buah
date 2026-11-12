# Sistem Dataset Buah

## Struktur dan jumlah kelas

Daftar pada `src/deteksi_buah.py` berisi **30 entri**, walaupun permintaan menyebut 29. `cantaloup/cantaloupe` adalah satu kelas alias, bukan dua kelas. Folder menggunakan ID yang aman untuk nama berkas: `cantaloupe`, `coconut`, `date`, `fig`, `kiwi_fruit`, dan seterusnya. Struktur sumbernya:

```text
dataset/<fruit_id>/{segar,kurang_segar,busuk}/
```

Gambar lama seperti `dataset/pisang1.png` dan folder `dataset/segar`/`dataset/busuk` di root dipertahankan, tetapi **tidak dimasukkan otomatis** karena belum ada metadata yang memastikan buah dan kesegarannya.

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

Saat pipeline disiapkan, project hanya memiliki direktori `dataset/segar` dan `dataset/busuk` yang kosong serta satu `dataset/pisang1.png` tanpa label yang dapat diverifikasi. Karena itu tidak ada angka laporan awal yang dapat dianggap berlabel benar dan belum ada dataset segar/kurang segar/busuk yang dapat dipakai latih. Sumber publik, lisensi, penandaan kesegaran, dan anotasi bbox harus dikurasi sebelum training.
