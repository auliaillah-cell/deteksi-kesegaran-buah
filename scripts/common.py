"""Konstanta dan utilitas untuk pengelolaan dataset buah."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
SPLIT_ROOT = ROOT / "dataset_split"
FRUITS = [
    ("almond", "Almond"), ("apple", "Apel"), ("apricot", "Aprikot"),
    ("avocado", "Alpukat"), ("banana", "Pisang"), ("blackberry", "Blackberry"),
    ("blueberry", "Blueberry"), ("cantaloupe", "Melon"), ("cherry", "Ceri"),
    ("clementine", "Clementine"), ("coconut", "Kelapa"), ("date", "Kurma"),
    ("fig", "Ara"), ("grape", "Anggur"), ("kiwi_fruit", "Kiwi"),
    ("lemon", "Lemon"), ("lime", "Jeruk nipis"),
    ("mandarin_orange", "Jeruk mandarin"), ("melon", "Melon"),
    ("orange", "Jeruk"), ("papaya", "Pepaya"), ("peach", "Persik"),
    ("pear", "Pir"), ("persimmon", "Kesemek"), ("pineapple", "Nanas"),
    ("prune", "Plum"), ("raspberry", "Raspberry"),
    ("strawberry", "Stroberi"), ("tomato", "Tomat"), ("watermelon", "Semangka"),
]
FRESHNESS = ("segar", "kurang_segar", "busuk")
SPLITS = (("train", .70), ("val", .20), ("test", .10))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def image_files(directory):
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
