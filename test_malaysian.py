import os
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead

# --- 1. SETUP ---
paddle.set_device('cpu')
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_PATH = "malaysian_ssd_BEST.pth"
LABEL_MAP = {1: 'PLATE', 2: 'CAR', 3: 'LOGO', 4: 'BADGE'}
COLORS = {1: "red", 2: "blue", 3: "cyan", 4: "green"}

# Load Model
model = ssd300_vgg16(weights=None)
model.head = SSDHead([512, 1024, 512, 256, 256, 256], [4, 6, 6, 6, 4, 4], 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

# --- 2. IMAGE PREP ---
img_path = "dataset/test/IMG_6009.jpg"  # Update this for your different car data!
if not os.path.exists(img_path):
    print(f"ERROR: Image {img_path} not found!")
    exit()

img_pil = Image.open(img_path).convert("RGB")
cv_img = cv2.imread(img_path)
w, h = img_pil.size
img_tensor = torchvision.transforms.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)

f_size = int(h * 0.025)
try:
    font = ImageFont.truetype("arial.ttf", f_size)
except:
    font = ImageFont.load_default()

# --- 3. INFERENCE ---
print("-" * 60)
print(f"STARTING RECOGNITION ON: {img_path}")
print("-" * 60)

with torch.no_grad():
    prediction = model(img_tensor)

draw = ImageDraw.Draw(img_pil)

# --- 4. THE RECOGNITION & SMART DRAWING LOOP ---
found_anything = False

for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
    s = score.item()
    l_idx = label.item()

    # --- DYNAMIC THRESHOLD ---
    current_threshold = 0.01 if l_idx in [3, 4] else 0.12

    if s > current_threshold:
        found_anything = True
        l_name = LABEL_MAP.get(l_idx, "UNKNOWN")
        b = box.tolist()

        # --- OCR FOR TEXT ---
        ocr_text = ""
        if l_idx in [1, 3, 4]:  # Read text for Plates, Logos, or Badges
            x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
            crop = cv_img[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                result = ocr_model.ocr(crop, cls=True)
                if result and result[0]:
                    ocr_text = result[0][0][1][0].upper()

        # Define the label text to be used in console and on image
        label_display = f"{l_name}: {ocr_text}" if ocr_text else f"{l_name} ({s:.2f})"

        # --- CONSOLE PRINTING ---
        status = f"[{l_name}]"
        print(f"FOUND: {status:<10} | Confidence: {s:.4f} | Text: {ocr_text if ocr_text else '(No Text Detected)'}")

        # --- SMART DRAWING ON IMAGE ---
        color = COLORS.get(l_idx, "white")
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=5)

        # UI Logic: Draw Plate labels BELOW, others ABOVE
        # This solves the overlapping problem you saw!
        if l_idx == 1: # PLATE
            text_y = b[3] + 5 # Start text below the bottom of the box
            bg_rect = [b[0], b[3], b[0] + (len(label_display) * (f_size * 0.6)), b[3] + f_size + 10]
        else: # LOGO / BADGE / CAR
            text_y = b[1] - (f_size + 10) # Start text above the top of the box
            bg_rect = [b[0], b[1] - (f_size + 10), b[0] + (len(label_display) * (f_size * 0.6)), b[1]]

        # Draw the background colored box and then the white text
        draw.rectangle(bg_rect, fill=color)
        draw.text((b[0] + 5, text_y), label_display, fill="white", font=font)


if not found_anything:
    print("No objects detected above the threshold.")

print("-" * 60)
print("SUCCESS: Check 'recognition_result.jpg' for visual proof.")
print("-" * 60)

img_pil.save("recognition_result.jpg")
img_pil.show()