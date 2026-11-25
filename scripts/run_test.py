import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src.deteksi_buah import deteksi_buah

img_path = r"C:/Users/ASUS/.gemini/antigravity/brain/8f9f01b0-ea13-4c13-88a1-8ccbd71c295f/.user_uploaded/media_1795536405993_e3b32a5f.png"
try:
    annotated, detections = deteksi_buah(img_path)
    print('Deteksi selesai, buah ditemukan:', len(detections))
    for i, b in enumerate(detections, 1):
        print(i, b)
except Exception as e:
    print('Error during detection:', e)
