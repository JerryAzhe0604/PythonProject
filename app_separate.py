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
st.set_page_config(page_title="Malaysian Car AI - Specialist Mode", layout="wide")
st.title("🚗 Specialist Model Overlay System")
st.write("Running Car, Brand, and Plate models simultaneously for maximum precision.")

# --- CONSTANTS ---
CAR_MODEL_PATH = "car_specialist.pth"
BRAND_MODEL_PATH = "brand_specialist.pth"
PLATE_MODEL_PATH = "plate_specialist.pth"

LABEL_MAP = {
    1: 'PLATE', 2: 'CAR', 3: 'PERODUA', 4: 'PROTON', 5: 'HONDA',
    6: 'TOYOTA', 7: 'MERCEDES', 8: 'BMW', 9: 'NISSAN', 10: 'OTHERS'
}


# --- COLOR DETECTION HELPER ---
def detect_car_color(img_cv, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
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


# --- THE SYNCED ARCHITECTURE (Must match training) ---
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


# --- TRIPLE MODEL LOADING ---
@st.cache_resource
def load_all_specialists():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=False, show_log=False)

    # Load Car Model
    car_model = create_model(11)
    if os.path.exists(CAR_MODEL_PATH):
        car_model.load_state_dict(torch.load(CAR_MODEL_PATH, map_location=device), strict=False)
    car_model.to(device).eval()

    # Load Brand Model
    brand_model = create_model(11)
    if os.path.exists(BRAND_MODEL_PATH):
        brand_model.load_state_dict(torch.load(BRAND_MODEL_PATH, map_location=device), strict=False)
    brand_model.to(device).eval()

    # Load Plate Model
    plate_model = create_model(11)
    if os.path.exists(PLATE_MODEL_PATH):
        plate_model.load_state_dict(torch.load(PLATE_MODEL_PATH, map_location=device), strict=False)
    plate_model.to(device).eval()

    return car_model, brand_model, plate_model, ocr, device


car_net, brand_net, plate_net, ocr_model, DEVICE = load_all_specialists()

# --- SIDEBAR ---
st.sidebar.header("Individual Thresholds")
t_car = st.sidebar.slider("Car Model Sensitivity", 0.0, 1.0, 0.40)
t_brand = st.sidebar.slider("Brand Model Sensitivity", 0.0, 1.0, 0.20)
t_plate = st.sidebar.slider("Plate Model Sensitivity", 0.0, 1.0, 0.30)

file = st.file_uploader("Upload Malaysian Car Image", type=["jpg", "png", "jpeg"])

if file:
    img_pil = Image.open(file).convert("RGB")
    cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    w, h = img_pil.size
    draw = ImageDraw.Draw(img_pil)

    # Pre-processing (One time for all three models)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(img_pil).unsqueeze(0).to(DEVICE)

    # --- TRIPLE INFERENCE ---
    with torch.no_grad():
        out_car = car_net(img_t)[0]
        out_brand = brand_net(img_t)[0]
        out_plate = plate_net(img_t)[0]

    x_scale, y_scale = w / 512, h / 512
    line_w = max(4, int(h / 150))
    f_size = max(20, int(h / 35))
    try:
        font = ImageFont.truetype("arial.ttf", f_size)
    except:
        font = ImageFont.load_default()

    detected_list = []


    # Helper to scale boxes
    def scale_box(box):
        b = box.cpu().numpy()
        return [b[0] * x_scale, b[1] * y_scale, b[2] * x_scale, b[3] * y_scale]


    # 1. PROCESS BRAND SPECIALIST RESULTS
    brands_found = []
    for box, label, score in zip(out_brand['boxes'], out_brand['labels'], out_brand['scores']):
        if score > t_brand and label.item() >= 3:
            sb = scale_box(box)
            name = LABEL_MAP[label.item()]
            brands_found.append({'box': sb, 'name': name})
            draw.rectangle(sb, outline="yellow", width=int(line_w / 2))
            draw.text((sb[0], sb[1] - f_size), name, font=font, fill="yellow")

    # 2. PROCESS CAR SPECIALIST RESULTS
    for box, label, score in zip(out_car['boxes'], out_car['labels'], out_car['scores']):
        if score > t_car and label.item() == 2:
            cb = scale_box(box)
            # Find matching brand within this car box
            car_brand = "Unknown"
            for b in brands_found:
                bb = b['box']
                if cb[0] < (bb[0] + bb[2]) / 2 < cb[2] and cb[1] < (bb[1] + bb[3]) / 2 < cb[3]:
                    car_brand = b['name']
                    break

            color = detect_car_color(cv_img, cb)
            label_text = f"{color} {car_brand} CAR"
            draw.rectangle(cb, outline="#00FF00", width=line_w)
            draw.text((cb[0] + 5, cb[1] + 5), label_text, font=font, fill="#00FF00")
            detected_list.append({"Type": "Vehicle", "Conf": f"{score:.1%}", "Detail": label_text})

    # 3. PROCESS PLATE SPECIALIST RESULTS
    for box, label, score in zip(out_plate['boxes'], out_plate['labels'], out_plate['scores']):
        if score > t_plate and label.item() == 1:
            pb = scale_box(box)
            crop = cv_img[max(0, int(pb[1])):min(h, int(pb[3])), max(0, int(pb[0])):min(w, int(pb[2]))]
            txt = "PLATE"
            if crop.size > 0:
                res = ocr_model.ocr(crop, cls=True)
                if res and res[0]: txt = res[0][0][1][0].upper()

            draw.rectangle(pb, outline="#FF0000", width=line_w)
            draw.text((pb[0], pb[3] + 5), txt, font=font, fill="#FF0000")
            detected_list.append({"Type": "Plate", "Conf": f"{score:.1%}", "Detail": txt})

    # UI DISPLAY
    c1, c2 = st.columns([3, 1])
    with c1:
        st.image(img_pil, use_container_width=True)
    with c2:
        st.dataframe(detected_list, use_container_width=True)