import os
import shutil
import random

# 1. Setup Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
source_folder = os.path.join(base_dir, 'dataset', 'ccpd_base')
train_folder = os.path.join(base_dir, 'dataset', 'train')
test_folder = os.path.join(base_dir, 'dataset', 'test')

# --- THE "DELETE" SECTION ---
# This ONLY clears the target folders to keep the split fresh.
# It does NOT touch your source folder.
for folder in [train_folder, test_folder]:
    if os.path.exists(folder):
        print(f"Cleaning out old data in: {folder}...")
        shutil.rmtree(folder)
    os.makedirs(folder)

# 2. Safety Check: Make sure your 1,000 images are actually there
if not os.path.exists(source_folder) or len(os.listdir(source_folder)) == 0:
    print(f"!!! ERROR: {source_folder} is empty. Put your 1,000 images in there first!")
    exit()

# 3. Grab the images
all_images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png'))]
random.seed(42) # Keeps the "random" split the same every time
random.shuffle(all_images)

# We take exactly 1,000 images (or fewer if that's all you have)
selected_images = all_images[:1000]

# 4. Calculate 70/30 Split
split_idx = int(len(selected_images) * 0.7)
train_files = selected_images[:split_idx] # 700 images
test_files = selected_images[split_idx:]  # 300 images

# 5. COPY (Safety First!)
print(f"Copying {len(train_files)} images to Train (70%)...")
for f in train_files:
    shutil.copy(os.path.join(source_folder, f), os.path.join(train_folder, f))

print(f"Copying {len(test_files)} images to Test (30%)...")
for f in test_files:
    shutil.copy(os.path.join(source_folder, f), os.path.join(test_folder, f))

print("\n" + "="*30)
print("SPLIT COMPLETE")
print(f"Source: {len(all_images)} available")
print(f"Train:  {len(train_files)} (70%)")
print(f"Test:   {len(test_files)} (30%)")
print("="*30)