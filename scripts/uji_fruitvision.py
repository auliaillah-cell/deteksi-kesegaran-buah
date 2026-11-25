import os
import sys

# Ensure project root is on PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.deteksi_buah import deteksi_buah, _hitung_total_buah

# Paths to the two sample images (uploaded by the user)
image_dir = r"C:/Users/ASUS/.gemini/antigravity/brain/8f9f01b0-ea13-4c13-88a1-8ccbd71c295f/.user_uploaded"
images = [
    os.path.join(image_dir, "media_1795504160996_a88eac8d.png"),
    os.path.join(image_dir, "media_1795504185746_f9613ce9.jpg"),
]

for idx, img_path in enumerate(images, start=1):
    print(f"\n=== Testing image {idx}: {os.path.basename(img_path)} ===", flush=True)
    try:
        annotated_img, detections = deteksi_buah(img_path)
    except Exception as e:
        print(f"Error during detection: {e}", flush=True)
        continue

    # Print per‑fruit details (already printed inside deteksi_buah, but repeat summary)
    totals = _hitung_total_buah(detections)
    print("\n--- Summary ---", flush=True)
    print(f"Total buah terdeteksi: {totals['total']}", flush=True)
    print(f"  Segar       : {totals['segar']}", flush=True)
    print(f"  Kurang segar: {totals['kurang_segar']}", flush=True)
    print(f"  Busuk       : {totals['busuk']}", flush=True)
    print("--- End of image ---\n", flush=True)
