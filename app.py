import streamlit as st
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaysian Car Recognition AI", layout="wide")
st.title("🚗 Malaysian Car Recognition System (SSD300)")
st.write("Upload an image of a car to detect Plates, Logos, and Badges.")

# --- CONSTANTS ---
MODEL_PATH = "malaysian_ssd_BEST.pth"
LABEL_MAP = {1: 'PLATE', 2: 'CAR', 3: 'LOGO', 4: 'BADGE'}
COLORS = {1: "red", 2: "blue", 3: "cyan", 4: "green"}


# --- MODEL LOADING (CACHED) ---
@st.cache_resource
def load_models():
    # 1. Load PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)

    # 2. Load SSD300
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = ssd300_vgg16(weights=None)
    model.head = SSDHead([512, 1024, 512, 256, 256, 256], [4, 6, 6, 6, 4, 4], 5)

    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    return ocr, model, device


ocr_model, detection_model, DEVICE = load_models()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Settings")
threshold_plate = st.sidebar.slider("Plate/Car Threshold", 0.0, 1.0, 0.15)
threshold_logo = st.sidebar.slider("Logo/Badge Threshold", 0.0, 1.0, 0.05)

# --- UPLOAD SECTION ---
uploaded_file = st.file_uploader("Choose a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Prepare Image
    image = Image.open(uploaded_file).convert("RGB")
    cv_img = np.array(image)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    w, h = image.size

    # 2. Run Inference
    img_tensor = torchvision.transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        prediction = detection_model(img_tensor)

    # 3. Process Results
    draw = ImageDraw.Draw(image)
    f_size = int(h * 0.025)
    try:
        font = ImageFont.truetype("arial.ttf", f_size)
    except:
        font = ImageFont.load_default()

    st.subheader("Recognition Results")
    col1, col2 = st.columns([3, 1])

    detected_objects = []

    for box, label, score in zip(prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']):
        s = score.item()
        l_idx = label.item()

        # Use Dynamic Thresholds from sidebar
        current_threshold = threshold_logo if l_idx in [3, 4] else threshold_plate

        if s > current_threshold:
            l_name = LABEL_MAP.get(l_idx, "UNKNOWN")
            b = box.tolist()

            # OCR Logic
            ocr_text = ""
            if l_idx in [1, 3, 4]:
                x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(w, int(b[2])), min(h, int(b[3]))
                crop = cv_img[y1:y2, x1:x2]
                if crop.size > 0:
                    res = ocr_model.ocr(crop, cls=True)
                    if res and res[0]:
                        ocr_text = res[0][0][1][0].upper()

            # Drawing
            color = COLORS.get(l_idx, "white")
            label_display = f"{l_name}: {ocr_text}" if ocr_text else f"{l_name} ({s:.2f})"

            draw.rectangle([b[0], b[1], b[2], b[3]], outline=color, width=5)

            # Smart Drawing (Plates below, others above)
            if l_idx == 1:
                text_y = b[3] + 5
                bg_rect = [b[0], b[3], b[0] + (len(label_display) * (f_size * 0.6)), b[3] + f_size + 10]
            else:
                text_y = b[1] - (f_size + 10)
                bg_rect = [b[0], b[1] - (f_size + 10), b[0] + (len(label_display) * (f_size * 0.6)), b[1]]

            draw.rectangle(bg_rect, fill=color)
            draw.text((b[0] + 5, text_y), label_display, fill="white", font=font)

            detected_objects.append({"Type": l_name, "Confidence": f"{s:.2%}", "Text": ocr_text})

    # 4. Display Results
    col1.image(image, use_container_width=True)
    if detected_objects:
        col2.dataframe(detected_objects)
    else:
        col2.write("No objects detected.")