import unittest

from src.deteksi_buah import _id_kelas_buah, _tambahkan_deteksi_unik


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


if __name__ == "__main__":
    unittest.main()
