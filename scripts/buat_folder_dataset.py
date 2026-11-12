from common import DATASET, FRUITS, FRESHNESS

def main():
    for fruit, _ in FRUITS:
        for condition in FRESHNESS:
            (DATASET / fruit / condition).mkdir(parents=True, exist_ok=True)
    print(f"Struktur dibuat untuk {len(FRUITS)} buah dan {len(FRESHNESS)} kategori.")

if __name__ == "__main__":
    main()
