import os
import xml.etree.ElementTree as ET


def update_tags_for_brands():
    base_path = 'dataset/sorted_train'
    brand_mapping = {
        'perodua': 'logo_perodua',
        'proton': 'logo_proton',
        'honda': 'logo_honda',
        'toyota': 'logo_toyota',
        'mercedes': 'logo_mercedes',
        'bmw': 'logo_bmw',
        'nissan': 'logo_nissan',
        'others': 'logo_others'
    }

    for folder, new_tag in brand_mapping.items():
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path): continue

        print(f"Updating XMLs in: {folder}...")
        for filename in os.listdir(folder_path):
            if filename.endswith('.xml'):
                file_path = os.path.join(folder_path, filename)
                tree = ET.parse(file_path)
                root = tree.getroot()

                for obj in root.findall('object'):
                    name = obj.find('name')
                    if name.text.lower() in ['logo', 'model_badge']:
                        name.text = new_tag

                tree.write(file_path)
    print("All XML tags updated successfully!")


if __name__ == "__main__":
    update_tags_for_brands()