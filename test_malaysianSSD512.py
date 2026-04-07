import os
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
import paddle
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from torchvision.models.vgg import vgg16

# --- 1. SETUP ---
# Initialize PaddleOCR for Malaysian plate/badge reading
paddle.set_device('cpu')
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# UPDATE: Point this to your teammate's SSD512 model file
MODEL_PATH = "malaysian_ssd512_allinone.pth"

LABEL_MAP = {1: 'PLATE', 2: 'CAR', 3: 'LOGO', 4: 'BADGE'}
COLORS = {1: "red", 2: "blue", 3: "cyan", 4: "green"}


# --- SSD512 ARCHITECTURE CONSTRUCTION ---
# This must match the training script exactly to load the .pth weights
def create_ssd512(num_classes):
    # Base VGG16 features
    backbone_vgg = vgg16(weights=None).features

    # Extract layers for SSD (standard VGG16 SSD uses up to layer 30)
    backbone = torchvision.models.detection.backbone_utils.SpecifiableChildModulesLayer(
        backbone_vgg, [str(i) for i in range(30)]
    )

    # SSD512 Anchor Generator (7 feature maps vs 6 in SSD300)
    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )

    # Head configuration for 7 feature maps
    in_channels = [512, 1024, 512, 256, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 6, 4, 4]
    head = SSDHead(in_channels, num_anchors, num_classes)

    # Final SSD512 Assembly
    return SSD(
        backbone=backbone,
        anchor_generator=anchor_generator,
        size=(512, 512),
        head=head
    )


# Load Model structure (5 classes: background + 4 labels)
model = create_ssd512(num_classes=5)

# Load the weights
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"SUCCESS: Loaded {MODEL_PATH}")
else:
    print(f"WARNING: {MODEL_PATH} not found. Running with uninitialized weights!")

model.to(DEVICE).eval()

# --- 2. IMAGE PREP ---
img_path = "dataset/test/IMG_6005.jpg"  # Update this to test different cars
if not os.path.exists(img_path):
    print(f"ERROR: Image {img_path} not found!")
    exit()

img_pil = Image.open(img_path).convert("RGB")
cv_img = cv2.imread(img_path)
w, h = img_pil.size

# Torchvision handles the resize to 512x512 automatically via the model.size parameter
img_tensor = torchvision.transforms.ToTensor()(img_pil).unsqueeze(0).to(DEVICE)

f_size = int(h * 0.025)
try:
    font = ImageFont.truetype("arial.ttf", f_size)
except:
    font = ImageFont.load_default()

# --- 3. INFERENCE ---
print("-" * 60)
print(f"STARTING SSD512 RECOGNITION ON: {img_path}")
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
    # Logos and Badges are small, so we use a more sensitive threshold
    current_threshold = 0.05 if l_idx in [3, 4] else 0.15

    if s > current_threshold:
        found_anything = True
        l_name = LABEL_MAP.get(l_idx, "UNKNOWN")
        b = box.tolist()

        # --- OCR FOR TEXT ---
        ocr_text = ""
        if l_idx in [1, 3, 4]:  # Plates, Logos, or Badges
            x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
            crop = cv_img[y1:y2, x1:x2]
            if crop is not None and crop.size > 0:
                result = ocr_model.ocr(crop, cls=True)
                if result and result[0]:
                    ocr_text = result[0][0][1][0].upper()

        # Label formatting
        label_display = f"{l_name}: {ocr_text}" if ocr_text else f"{l_name} ({s:.2f})"

        # --- CONSOLE PRINTING ---
        print(f"FOUND: [{l_name:<6}] | Conf: {s:.4f} | Text: {ocr_text if ocr_text else '(None)'}")

        # --- SMART DRAWING ---
        color = COLORS.get(l_idx, "white")
        draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=5)

        # Plate labels go BELOW, others go ABOVE to prevent overlap
        if l_idx == 1:  # PLATE
            text_y = b[3] + 5
            bg_rect = [b[0], b[3], b[0] + (len(label_display) * (f_size * 0.6)), b[3] + f_size + 10]
        else:  # LOGO / BADGE / CAR
            text_y = b[1] - (f_size + 10)
            bg_rect = [b[0], b[1] - (f_size + 10), b[0] + (len(label_display) * (f_size * 0.6)), b[1]]

        draw.rectangle(bg_rect, fill=color)
        draw.text((b[0] + 5, text_y), label_display, fill="white", font=font)

if not found_anything:
    print("No objects detected above the threshold.")

print("-" * 60)
print("SUCCESS: Check 'recognition_result.jpg' for the SSD512 output.")
print("-" * 60)

img_pil.save("recognition_result.jpg")
img_pil.show()