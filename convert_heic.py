import os
from PIL import Image
from pillow_heif import register_heif_opener

# 1. Initialize the HEIF opener for Pillow
register_heif_opener()

# 2. Define your paths
base_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(base_dir, 'dataset', 'malaysian_raw')
output_folder = os.path.join(base_dir, 'dataset', 'malaysian_cars')

# 3. Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Created folder: {output_folder}")

# 4. Process the files
print(f"--- Starting Conversion: {input_folder} -> {output_folder} ---")

files = [f for f in os.listdir(input_folder) if f.lower().endswith(".heic")]

if len(files) == 0:
    print("No .HEIC files found! Double-check your 'malaysian_raw' folder.")
else:
    for i, filename in enumerate(files):
        try:
            heic_path = os.path.join(input_folder, filename)
            # Create the new filename by replacing the extension
            new_name = os.path.splitext(filename)[0] + ".jpg"
            jpg_path = os.path.join(output_folder, new_name)

            # Open, Convert to RGB (Required for JPG), and Save
            image = Image.open(heic_path)
            image = image.convert('RGB')
            image.save(jpg_path, "JPEG", quality=95)

            print(f"[{i + 1}/{len(files)}] Converted: {filename} -> {new_name}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("\nConversion process complete! Your JPGs are ready for labeling.")