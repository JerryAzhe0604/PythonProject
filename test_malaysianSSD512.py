import torch
import torchvision
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from paddleocr import PaddleOCR
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from torchvision.models.vgg import vgg16
import torch.nn as nn

def get_color(img_cv, box):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    crop = img_cv[y1+int(h*0.3):y1+int(h*0.7), x1+int(w*0.3):x1+int(w*0.7)]
    if crop.size == 0: return "Unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    avg_s, avg_v, avg_h = np.mean(hsv[:,:,1]), np.mean(hsv[:,:,2]), np.mean(hsv[:,:,0])
    if avg_v < 50: return "Black"
    if avg_s < 30 and avg_v > 180: return "White"
    if avg_s < 30: return "Silver"
    if 0 <= avg_h < 10 or 160 <= avg_h <= 180: return "Red"
    if 90 <= avg_h < 130: return "Blue"
    return "Colored"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

def create_ssd512(num_classes):
    backbone = nn.Sequential(*list(vgg16(weights=None).features)[:30])
    anchor_gen = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
                                     scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
                                     steps=[8, 16, 32, 64, 128, 256, 512])
    head = SSDHead([512, 1024, 512, 256, 256, 256, 256], [4, 6, 6, 6, 6, 4, 4], num_classes)
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)

# Must match training: 11
model = create_ssd512(num_classes=11)
model.load_state_dict(torch.load("malaysian_ssd512_RESCUE.pth", map_location=DEVICE))
model.to(DEVICE).eval()

LABEL_MAP = {
    1:'PLATE', 2:'CAR', 3:'PERODUA', 4:'PROTON', 5:'HONDA',
    6:'TOYOTA', 7:'MERCEDES', 8:'BMW', 9:'NISSAN', 10:'OTHERS'
}

img_path = "dataset/test/IMG_6002.jpg"
img_pil = Image.open(img_path).convert("RGB")
cv_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
img_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img_pil).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    preds = model(img_tensor)[0]

draw = ImageDraw.Draw(img_pil)
for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):
    if score > 0.15:
        b = box.tolist()
        l_idx = label.item()
        l_name = LABEL_MAP.get(l_idx, "OBJ")
        info = l_name
        if l_idx == 2: info = f"{get_color(cv_img, b)} {l_name}"
        elif l_idx == 1:
            crop = cv_img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
            res = ocr.ocr(crop)
            if res and res[0]: info = res[0][0][1][0]
        draw.rectangle(b, outline="red", width=3)
        draw.text((b[0], b[1]-15), info, fill="white")
img_pil.show()