import streamlit as st
import torch
import torch.nn as nn
import torchvision
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from torchvision.models.vgg import vgg16

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaysian Car Recognition AI", layout="wide")
st.title("🚗 Malaysian Car Recognition System (SSD512)")
st.write("Detecting Plates, Brands, and Colors using High-Res SSD512.")

# --- CONSTANTS (Synced with your training) ---
MODEL_PATH = "malaysian_ssd512_RESCUE.pth"
LABEL_MAP = {
    1: 'PLATE', 2: 'CAR', 3: 'PERODUA', 4: 'PROTON', 5: 'HONDA',
    6: 'TOYOTA', 7: 'MERCEDES', 8: 'BMW', 9: 'NISSAN', 10: 'OTHERS'
}


# --- COLOR DETECTION HELPER ---
def detect_car_color(img_cv, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    crop = img_cv[y1 + int(h * 0.3):y1 + int(h * 0.7), x1 + int(w * 0.3):x1 + int(w * 0.7)]
    if crop.size == 0: return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_s, avg_v, avg_h = np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2]), np.mean(hsv[:, :, 0])
    if avg_v < 50: return "Black"
    if avg_s < 30 and avg_v > 180: return "White"
    if avg_s < 30: return "Silver"
    if 0 <= avg_h < 10 or 160 <= avg_h <= 180: return "Red"
    if 90 <= avg_h < 130: return "Blue"
    return "Colored"


# --- SSD512 ARCHITECTURE ---
def create_ssd512(num_classes):
    backbone = nn.Sequential(*list(vgg16(weights=None).features)[:30])
    anchor_gen = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                                     scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
                                     steps=[8, 16, 32, 64, 128, 256, 512])
    head = SSDHead([512, 1024, 512, 256, 256, 256, 256], [4, 6, 6, 6, 6, 4, 4], num_classes)
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False, show_log=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Change to 11 classes (10 objects + 1 background)
    model = create_ssd512(num_classes=11)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device).eval()
    else:
        st.error(f"Weights file {MODEL_PATH} not found!")
    return ocr, model, device


ocr_model, detection_model, DEVICE = load_models()

# --- SIDEBAR ---
st.sidebar.header("Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.20)

# --- UPLOAD ---
uploaded_file = st.file_uploader("Upload Car Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    w, h = image.size

    # Inference
    img_t = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = detection_model(img_t)[0]

    # Filter detections
    cars, logos, plates = [], [], []
    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
        if score > conf_threshold:
            d = {'box': box.tolist(), 'label': label.item(), 'score': score.item()}
            if d['label'] == 1:
                plates.append(d)
            elif d['label'] == 2:
                cars.append(d)
            elif d['label'] >= 3:
                logos.append(d)

    draw = ImageDraw.Draw(image)
    results_data = []

    # 1. Process Cars + Brands
    for car in cars:
        cb = car['box']
        brand_text = ""
        for logo in logos:
            lb = logo['box']
            if cb[0] < (lb[0] + lb[2]) / 2 < cb[2] and cb[1] < (lb[1] + lb[3]) / 2 < cb[3]:
                brand_text = LABEL_MAP[logo['label']]
                break

        color = detect_car_color(cv_img, cb)
        label_text = f"{color} {brand_text} CAR".strip()
        draw.rectangle(cb, outline="green", width=5)
        draw.text((cb[0], cb[1] - 20), label_text, fill="white")
        results_data.append({"Object": "Vehicle", "Info": label_text})

    # 2. Process Plates + OCR
    for plate in plates:
        pb = plate['box']
        crop = cv_img[int(pb[1]):int(pb[3]), int(pb[0]):int(pb[2])]
        res = ocr_model.ocr(crop)
        plate_no = res[0][0][1][0] if res and res[0] else "Unknown Plate"

        draw.rectangle(pb, outline="red", width=3)
        draw.text((pb[0], pb[1] - 15), f"PLATE: {plate_no}", fill="yellow")
        results_data.append({"Object": "Plate", "Info": plate_no})

    # Display
    col1, col2 = st.columns([2, 1])
    col1.image(image, caption="Inference Output", use_container_width=True)
    col2.subheader("Analysis Data")
    col2.table(results_data)