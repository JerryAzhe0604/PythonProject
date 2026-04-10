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
from torchvision.ops import nms

torch.backends.cudnn.benchmark = True

# --- PAGE CONFIG ---
st.set_page_config(page_title="Malaysian Car AI - RSD2S3", layout="wide")
st.title("🚗 Malaysian Car Recognition System (SSD512)")

# --- CONSTANTS ---
MODEL_PATH = "malaysian_ssd512_improved.pth"

LABEL_MAP = {
    1: 'PLATE',
    2: 'CAR',
    3: 'PERODUA',
    4: 'PROTON',
    5: 'HONDA',
    6: 'TOYOTA',
    7: 'MERCEDES',
    8: 'BMW',
    9: 'NISSAN',
    10: 'OTHERS'
}


# --- COLOR DETECTION ---
def detect_car_color(img_cv, box):

    x1, y1, x2, y2 = map(int, box)

    h, w = y2 - y1, x2 - x1

    crop = img_cv[
        max(0, y1 + int(h * 0.3)):min(img_cv.shape[0], y1 + int(h * 0.7)),
        max(0, x1 + int(w * 0.3)):min(img_cv.shape[1], x1 + int(w * 0.7))
    ]

    if crop.size == 0:
        return "Unknown"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    avg_s = np.mean(hsv[:, :, 1])
    avg_v = np.mean(hsv[:, :, 2])
    avg_h = np.mean(hsv[:, :, 0])

    if avg_v < 50:
        return "Black"

    if avg_s < 30 and avg_v > 180:
        return "White"

    if avg_s < 30:
        return "Silver/Grey"

    if 0 <= avg_h < 10 or 160 <= avg_h <= 180:
        return "Red"

    if 90 <= avg_h < 130:
        return "Blue"

    return "Colored"


# --- SSD512 MODEL ---
def create_ssd512(num_classes):

    backbone = vgg16(weights=None).features

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )

    in_channels = [512, 1024, 512, 256, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 4, 4, 4]

    head = SSDHead(in_channels, num_anchors, num_classes)

    model = SSD(
        backbone,
        anchor_generator,
        (512, 512),
        num_classes,
        head=head
    )

    return model


# --- LOAD MODELS ---
@st.cache_resource
def load_models():

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        use_gpu=False,
        show_log=False
    )

    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    model = create_ssd512(num_classes=11)

    if os.path.exists(MODEL_PATH):

        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device),
            strict=False
        )

        model.to(device).eval()

        # Warmup
        dummy = torch.zeros((1, 3, 512, 512)).to(device)

        with torch.no_grad():
            model(dummy)

        st.sidebar.success("Model Loaded")

    return ocr, model, device


ocr_model, detection_model, DEVICE = load_models()

# --- SIDEBAR ---
st.sidebar.header("Settings")

threshold = st.sidebar.slider(
    "Confidence Threshold",
    0.0,
    1.0,
    0.25
)

# --- UPLOAD ---
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    cv_img = cv2.cvtColor(
        np.array(image),
        cv2.COLOR_RGB2BGR
    )

    w, h = image.size

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((512, 512)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    img_t = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = detection_model(img_t)[0]

    # NMS
    keep = nms(
        preds['boxes'],
        preds['scores'],
        0.45
    )

    boxes = preds['boxes'][keep]
    scores = preds['scores'][keep]
    labels = preds['labels'][keep]

    # Scale back
    scale_x = w / 512
    scale_y = h / 512

    line_thickness = max(4, int(h / 150))
    f_size = max(20, int(h / 30))

    try:
        font = ImageFont.truetype("arial.ttf", f_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(image)

    detected_data = []

    cars, logos, plates = [], [], []

    for box, label, score in zip(boxes, labels, scores):

        if score < threshold:
            continue

        box = box.cpu().numpy()

        box[0] *= scale_x
        box[2] *= scale_x
        box[1] *= scale_y
        box[3] *= scale_y

        d = {
            'box': box,
            'label': label.item(),
            'score': score.item()
        }

        if d['label'] == 1:
            plates.append(d)

        elif d['label'] == 2:
            cars.append(d)

        else:
            logos.append(d)

    # --- CAR DETECTION ---
    for car in cars:

        cb = car['box']
        brand = ""

        for logo in logos:

            lb = logo['box']

            if cb[0] < (lb[0] + lb[2]) / 2 < cb[2] and cb[1] < (lb[1] + lb[3]) / 2 < cb[3]:
                brand = LABEL_MAP[logo['label']]
                break

        color = detect_car_color(cv_img, cb)

        label_text = f"{color} {brand} CAR".strip()

        draw.rectangle(
            cb,
            outline="#00FF00",
            width=line_thickness
        )

        draw.rectangle(
            [cb[0], cb[1] - f_size - 10,
             cb[0] + len(label_text) * f_size * 0.6,
             cb[1]],
            fill="#00FF00"
        )

        draw.text(
            (cb[0] + 5, cb[1] - f_size - 8),
            label_text,
            font=font,
            fill="white"
        )

        detected_data.append({
            "Type": "Vehicle",
            "Conf": f"{car['score']:.1%}",
            "Info": label_text
        })

    # --- PLATE OCR ---
    for p in plates:

        pb = p['box']

        crop = cv_img[
            int(pb[1]):int(pb[3]),
            int(pb[0]):int(pb[2])
        ]

        plate_text = "PLATE"

        if crop.size > 0:

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            gray = cv2.resize(
                gray,
                None,
                fx=2,
                fy=2
            )

            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            res = ocr_model.ocr(
                gray,
                cls=True
            )

            if res and res[0]:
                plate_text = res[0][0][1][0].upper()

        draw.rectangle(
            pb,
            outline="#FF0000",
            width=line_thickness
        )

        draw.rectangle(
            [pb[0], pb[3],
             pb[0] + len(plate_text) * f_size * 0.6,
             pb[3] + f_size + 10],
            fill="#FF0000"
        )

        draw.text(
            (pb[0] + 5, pb[3] + 2),
            plate_text,
            font=font,
            fill="#FFFF00"
        )

        detected_data.append({
            "Type": "Plate",
            "Conf": f"{p['score']:.1%}",
            "Info": plate_text
        })

    # --- UI ---
    c1, c2 = st.columns([3, 1])

    c1.image(
        image,
        use_container_width=True
    )

    if detected_data:

        c2.dataframe(
            detected_data,
            use_container_width=True
        )

    else:

        max_score = (
            scores[0].item()
            if len(scores) > 0
            else 0
        )

        st.sidebar.warning(
            f"No detections. Highest score {max_score:.2%}"
        )