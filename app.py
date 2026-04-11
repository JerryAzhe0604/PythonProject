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
st.set_page_config(page_title="Malaysian Car AI - RSD2S3", layout="wide")
st.title("🚗 Malaysian Car Recognition System (SSD512)")
st.write("Detecting Color, Brand, and Plate with Individual Threshold Controls.")

# --- CONSTANTS ---
MODEL_PATH = "malaysian_ssd512_RESCUE.pth"
LABEL_MAP = {
    1: 'PLATE', 2: 'CAR', 3: 'PERODUA', 4: 'PROTON', 5: 'HONDA',
    6: 'TOYOTA', 7: 'MERCEDES', 8: 'BMW', 9: 'NISSAN', 10: 'OTHERS'
}


# --- COLOR DETECTION HELPER ---
def detect_car_color(img_cv, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    # ROI: Center body crop
    crop = img_cv[max(0, y1 + int(h * 0.3)):min(img_cv.shape[0], y1 + int(h * 0.7)),
    max(0, x1 + int(w * 0.3)):min(img_cv.shape[1], x1 + int(w * 0.7))]
    if crop.size == 0: return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_s, avg_v = np.mean(hsv[:, :, 1]), np.mean(hsv[:, :, 2])
    avg_h = np.mean(hsv[:, :, 0])

    if avg_v < 50: return "Black"
    if avg_s < 30 and avg_v > 180: return "White"
    if avg_s < 30: return "Silver/Grey"
    if 0 <= avg_h < 10 or 160 <= avg_h <= 180: return "Red"
    if 90 <= avg_h < 130: return "Blue"
    return "Colored"


# --- THE SYNCED ARCHITECTURE ---
def create_model(num_classes):
    backbone_vgg = vgg16(weights=None).features

    class BackboneWrapper(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = nn.Sequential(*list(features)[:30])

        def forward(self, x): return self.features(x)

    backbone = BackboneWrapper(backbone_vgg)
    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06, 1.20],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )
    in_channels = [512, 1024, 512, 256, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4, 4]
    head = SSDHead(in_channels, num_anchors, num_classes)
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


# --- MODEL LOADING ---
@st.cache_resource
def load():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model(11)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        st.sidebar.success(f"Model Synced: {MODEL_PATH}")
    model.to(device).eval()
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)
    return model, ocr, device


model, ocr_model, DEVICE = load()

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Threshold Adjustments")
t_car = st.sidebar.slider("Car Detection Threshold", 0.0, 1.0, 0.40)
t_plate = st.sidebar.slider("Plate Detection Threshold", 0.0, 1.0, 0.30)
t_brand = st.sidebar.slider("Brand Logo Threshold", 0.0, 1.0, 0.15)

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img_pil = Image.open(file).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    w, h = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    # Scaling and Fonts
    line_w = max(4, int(h / 150))
    f_size = max(20, int(h / 35))
    try:
        font = ImageFont.truetype("arial.ttf", f_size)
    except:
        font = ImageFont.load_default()

    # Inference
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_t)[0]

    x_scale, y_scale = w / 512, h / 512
    cars, logos, plates = [], [], []

    # Categorize and Filter
    for box, label, score in zip(output['boxes'], output['labels'], output['scores']):
        l_idx = label.item()
        s_val = score.item()

        # Determine which threshold to use
        target_threshold = t_car if l_idx == 2 else (t_plate if l_idx == 1 else t_brand)

        if s_val > target_threshold:
            b = box.cpu().numpy()
            b[0] *= x_scale;
            b[2] *= x_scale
            b[1] *= y_scale;
            b[3] *= y_scale
            d = {'box': b.tolist(), 'label': l_idx, 'score': s_val}

            if l_idx == 1:
                plates.append(d)
            elif l_idx == 2:
                cars.append(d)
            elif l_idx >= 3:
                logos.append(d)

    detected_list = []

    # 1. PROCESS LOGOS (Draw first so they are under the car text)
    for logo in logos:
        lb = logo['box']
        name = LABEL_MAP[logo['label']]
        draw.rectangle(lb, outline="yellow", width=int(line_w / 2))
        draw.text((lb[0], lb[1] - (f_size / 2)), name, font=font, fill="yellow")

    # 2. PROCESS CARS
    for car in cars:
        cb = car['box']
        brand_text = "Unknown"
        # Logic to find which logo is inside this car box
        for logo in logos:
            lb = logo['box']
            if cb[0] < (lb[0] + lb[2]) / 2 < cb[2] and cb[1] < (lb[1] + lb[3]) / 2 < cb[3]:
                brand_text = LABEL_MAP[logo['label']]
                break

        color = detect_car_color(cv_img, cb)
        display_name = f"{color} {brand_text} CAR"

        draw.rectangle(cb, outline="#00FF00", width=line_w)
        draw.rectangle([cb[0], cb[1] - f_size - 10, cb[0] + (len(display_name) * f_size * 0.6), cb[1]], fill="#00FF00")
        draw.text((cb[0] + 5, cb[1] - f_size - 8), display_name, font=font, fill="white")
        detected_list.append(
            {"Type": "Vehicle", "Conf": f"{car['score']:.1%}", "Brand": brand_text, "Color": color, "Extra Info": ""})

    # 3. PROCESS PLATES
    for p in plates:
        pb = p['box']
        crop = cv_img[max(0, int(pb[1])):min(h, int(pb[3])), max(0, int(pb[0])):min(w, int(pb[2]))]
        plate_no = "PLATE"
        if crop.size > 0:
            res = ocr_model.ocr(crop, cls=True)
            if res and res[0]: plate_no = res[0][0][1][0].upper()

        draw.rectangle(pb, outline="#FF0000", width=line_w)
        draw.rectangle([pb[0], pb[3], pb[0] + (len(plate_no) * f_size * 0.6), pb[3] + f_size + 10], fill="#FF0000")
        draw.text((pb[0] + 5, pb[3] + 2), plate_no, font=font, fill="#FFFF00")
        detected_list.append({"Type": "Plate", "Conf": f"{p['score']:.1%}", "Brand": "N/A", "Color": "N/A",
                              "Extra Info": f"Number: {plate_no}"})

    # Output Layout
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Recognition Results")
        st.image(img_pil, use_container_width=True)
    with col2:
        st.subheader("Data Summary")
        if detected_list:
            st.dataframe(detected_list, use_container_width=True)
        else:
            st.info("No detections. Adjust the sliders in the sidebar.")