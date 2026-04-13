import os
import cv2
import shutil

# 1. SETUP PATHS
source_dir = 'dataset/master_train'
target_base = 'dataset/sorted_train'

# 2. DEFINE YOUR KEYS (s added for Nissan)
keys = {
    ord('p'): 'perodua',
    ord('r'): 'proton',
    ord('h'): 'honda',
    ord('t'): 'toyota',
    ord('m'): 'mercedes',
    ord('b'): 'bmw',
    ord('s'): 'nissan',
    ord('o'): 'others',
    ord('n'): 'no_logo'
}

for brand in keys.values():
    os.makedirs(os.path.join(target_base, brand), exist_ok=True)

files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]

print("--- MALAYSIAN CAR SORTER ACTIVE ---")
print("CONTROLS: p=Perodua, r=Proton, h=Honda, t=Toyota, m=Mercedes, b=BMW, s=Nissan, n=No Logo, ESC=Quit")
print(f"Total images: {len(files)}")

for f in files:
    img_path = os.path.join(source_dir, f)
    xml_path = img_path.replace('.jpg', '.xml')

    img = cv2.imread(img_path)
    if img is None: continue

    cv2.imshow('Sorter', cv2.resize(img, (800, 600)))
    key = cv2.waitKey(0)

    if key == 27: break  # ESC

    if key in keys:
        brand = keys[key]
        shutil.move(img_path, os.path.join(target_base, brand, f))
        if os.path.exists(xml_path):
            shutil.move(xml_path, os.path.join(target_base, brand, f.replace('.jpg', '.xml')))
        print(f"Moved {f} to {brand}")

cv2.destroyAllWindows()