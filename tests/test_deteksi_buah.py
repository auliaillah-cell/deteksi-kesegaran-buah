import unittest

from src.deteksi_buah import (
    NAMA_BUAH_TIDAK_TERIDENTIFIKASI,
    _gabungkan_deteksi_jeruk,
    _gabungkan_kotak_lintas_kelas,
    _id_kelas_buah,
    _tambahkan_deteksi_unik,
)


class DeteksiBuahTests(unittest.TestCase):
    def test_memilih_semua_id_untuk_nama_kelas_duplikat(self):
        nama_kelas = {
            30: "Strawberry",
            35: "Tomato",
            57: "strawberry",
            59: "tomato",
            62: "zucchini",
        }

        self.assertEqual(
            _id_kelas_buah(nama_kelas),
            [30, 35, 57, 59],
        )

    def test_deteksi_sekelas_yang_tumpang_tindih_hanya_dihitung_sekali(self):
        daftar_buah = [{
            "nama": "Stroberi",
            "koordinat": [10, 10, 50, 50],
            "keyakinan_deteksi": 0.7,
        }]
        deteksi_lebih_yakin = {
            "nama": "Stroberi",
            "koordinat": [11, 11, 51, 51],
            "keyakinan_deteksi": 0.9,
        }

        _tambahkan_deteksi_unik(daftar_buah, deteksi_lebih_yakin)

        self.assertEqual(daftar_buah, [deteksi_lebih_yakin])

    def test_deteksi_berbeda_tidak_dihapus_hanya_karena_tumpang_tindih(self):
        daftar_buah = [{
            "nama": "Ceri",
            "koordinat": [10, 10, 50, 50],
            "keyakinan_deteksi": 0.7,
        }]
        deteksi_buah_lain = {
            "nama": "Tomat",
            "koordinat": [11, 11, 51, 51],
            "keyakinan_deteksi": 0.9,
        }

        _tambahkan_deteksi_unik(daftar_buah, deteksi_buah_lain)

        self.assertEqual(len(daftar_buah), 2)

    def test_deteksi_jeruk_tidak_mengganti_kelas_buah_lain(self):
        daftar_buah = [{
            "nama": "Apel",
            "koordinat": [10, 10, 50, 50],
            "keyakinan_deteksi": 0.8,
        }]
        deteksi_jeruk = {
            "nama": "Jeruk",
            "koordinat": [11, 11, 51, 51],
            "keyakinan_deteksi": 0.9,
        }

        _gabungkan_deteksi_jeruk(daftar_buah, deteksi_jeruk)

        self.assertEqual(daftar_buah[0]["nama"], "Apel")
        self.assertEqual(len(daftar_buah), 1)

    def test_deteksi_jeruk_memperbarui_satu_kotak_mandarin(self):
        daftar_buah = [{
            "nama": "Jeruk mandarin",
            "koordinat": [10, 10, 50, 50],
            "keyakinan_deteksi": 0.6,
        }]
        deteksi_jeruk = {
            "nama": "Jeruk",
            "koordinat": [11, 11, 51, 51],
            "keyakinan_deteksi": 0.9,
        }

        _gabungkan_deteksi_jeruk(daftar_buah, deteksi_jeruk)

        self.assertEqual(daftar_buah, [deteksi_jeruk])

    def test_deteksi_jeruk_menyatukan_dua_kotak_citrus_yang_sama(self):
        daftar_buah = [
            {
                "nama": "Jeruk",
                "koordinat": [10, 10, 50, 50],
                "keyakinan_deteksi": 0.7,
            },
            {
                "nama": "Jeruk mandarin",
                "koordinat": [11, 11, 51, 51],
                "keyakinan_deteksi": 0.6,
            },
        ]
        deteksi_jeruk = {
            "nama": "Jeruk",
            "koordinat": [10, 10, 50, 50],
            "keyakinan_deteksi": 0.9,
        }

        _gabungkan_deteksi_jeruk(daftar_buah, deteksi_jeruk)

        self.assertEqual(daftar_buah, [deteksi_jeruk])

    def test_deteksi_jeruk_di_lokasi_baru_ditambahkan(self):
        daftar_buah = []
        deteksi_jeruk = {
            "nama": "Jeruk",
            "koordinat": [10, 10, 50, 50],
            "keyakinan_deteksi": 0.9,
        }

        _gabungkan_deteksi_jeruk(daftar_buah, deteksi_jeruk)

        self.assertEqual(daftar_buah, [deteksi_jeruk])

    def test_kotak_lintas_kelas_digabung_dan_kelas_ditandai_ambigu(self):
        daftar_buah = [
            {
                "nama": "Apel",
                "koordinat": [10, 10, 50, 50],
                "keyakinan_deteksi": 0.8,
            },
            {
                "nama": "Persik",
                "koordinat": [11, 11, 51, 51],
                "keyakinan_deteksi": 0.79,
            },
        ]

        hasil = _gabungkan_kotak_lintas_kelas(daftar_buah)

        self.assertEqual(len(hasil), 1)
        self.assertEqual(hasil[0]["nama"], NAMA_BUAH_TIDAK_TERIDENTIFIKASI)
        self.assertEqual(hasil[0]["kelas_alternatif"], ["Apel", "Persik"])

    def test_kotak_yang_mencakup_dua_buah_tidak_menggabungkan_keduanya(self):
        daftar_buah = [
            {
                "nama": "Pisang",
                "koordinat": [0, 0, 100, 100],
                "keyakinan_deteksi": 0.9,
            },
            {
                "nama": "Apel",
                "koordinat": [30, 0, 130, 100],
                "keyakinan_deteksi": 0.85,
            },
            {
                "nama": "Pir",
                "koordinat": [15, 0, 115, 100],
                "keyakinan_deteksi": 0.8,
            },
        ]

        hasil = _gabungkan_kotak_lintas_kelas(daftar_buah)

        self.assertEqual(len(hasil), 2)
        self.assertEqual(
            {buah["nama"] for buah in hasil},
            {"Pisang", "Apel"},
        )


if __name__ == "__main__":
    unittest.main()
