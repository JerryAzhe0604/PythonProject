import os
import xml.etree.ElementTree as ET

# Run this on all your master folders
DATA_FOLDERS = ["dataset/master_train", "dataset/master_valid", "dataset/master_test"]


def clean_for_fyp_map(folder):
    if not os.path.exists(folder):
        print(f"Skipping {folder} (not found)")
        return

    count = 0
    for filename in os.listdir(folder):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            for obj in root.findall('object'):
                name_tag = obj.find('name')
                # .strip() removes any hidden spaces or \n characters
                old_name = name_tag.text.lower().strip()

                # 1. PLATE MAPPING
                if 'plate' in old_name or 'licence' in old_name:
                    name_tag.text = 'plate'

                # 2. CAR MAPPING
                elif old_name in ['car', 'vehicle', 'auto', 'suv', 'truck', 'van']:
                    name_tag.text = 'car'

                # 3. LOGO MAPPING (Expanded brand list)
                elif any(brand in old_name for brand in
                         ['honda', 'proton', 'perodua', 'toyota', 'nissan', 'mazda', 'mercedes', 'bmw', 'volkswagen',
                          'logo']):
                    name_tag.text = 'logo'

                # 4. BADGE MAPPING
                elif any(m in old_name for m in ['myvi', 'x50', 'x70', 'bezza', 'saga', 'badge', 'vvt']):
                    name_tag.text = 'model_badge'

                else:
                    # This will help us find if there are any weird labels left
                    print(f"Warning: Found unhandled label '{old_name}' in {filename}")

            tree.write(file_path)
            count += 1
    print(f"--- Finished cleaning {count} files in {folder} ---")


for folder in DATA_FOLDERS:
    clean_for_fyp_map(folder)