import os
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image, ImageOps
import xml.etree.ElementTree as ET
import torchvision
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from torchvision.models.vgg import vgg16, VGG16_Weights
from torchvision import transforms as T


# ==========================================
# 1. DATASET (Scaled to 300x300)
# ==========================================
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, roots, label_map, transforms=None):
        self.roots = roots if isinstance(roots, list) else [roots]
        self.transforms = transforms
        self.label_map = label_map
        self.imgs = []

        for root in self.roots:
            if os.path.exists(root):
                for r, d, f in os.walk(root):
                    for file in f:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.imgs.append(os.path.join(r, file))

        print(f"--- Dataset Ready: {len(self.imgs)} images loaded ---")

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        xml_path = img_path.rsplit('.', 1)[0] + '.xml'

        if not os.path.exists(xml_path):
            return self.__getitem__((idx + 1) % len(self))

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.autocontrast(img)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes, labels = [], []

            for obj in root.findall('object'):
                name = obj.find('name').text.lower().strip()
                if name in self.label_map:
                    bnd = obj.find('bndbox')
                    xmin, ymin = float(bnd.find('xmin').text), float(bnd.find('ymin').text)
                    xmax, ymax = float(bnd.find('xmax').text), float(bnd.find('ymax').text)
                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(self.label_map[name])

            if len(boxes) == 0:
                return self.__getitem__((idx + 1) % len(self))

        except:
            return self.__getitem__((idx + 1) % len(self))

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        w, h = img.size
        # --- RESIZE TO 300x300 ---
        img = img.resize((300, 300))
        if boxes.shape[0] > 0:
            target["boxes"][:, [0, 2]] *= (300.0 / w)
            target["boxes"][:, [1, 3]] *= (300.0 / h)

        img = T.ToTensor()(img)
        if self.transforms:
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


# ==========================================
# 2. SSD300 ARCHITECTURE
# ==========================================
def create_ssd300(num_classes):
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

    class BackboneWrapper(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = nn.Sequential(*list(features)[:30])

        def forward(self, x): return self.features(x)

    backbone = BackboneWrapper(vgg)

    # SSD300 uses 6 detection layers (Standard configuration)
    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05],
        steps=[8, 16, 32, 64, 100, 300]
    )

    # Head must match the 6 layers of anchor_gen
    head = SSDHead([512, 1024, 512, 256, 256, 256], [4, 6, 6, 6, 4, 4], num_classes)

    return SSD(backbone, anchor_gen, (300, 300), num_classes, head=head)


# ==========================================
# 3. TRAINING LOOP
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_ssd300(num_classes=11).to(DEVICE)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler('cuda')

    train_folders = [r"Car Models.v2-carobject.voc/train"]
    l_map = {'cars': 2, 'car': 2, 'vehicle': 2}

    dataset = XMLDataset(train_folders, l_map, transforms=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                         collate_fn=lambda b: tuple(zip(*b)), num_workers=0)

    print(f"--- STARTING SSD300 SPECIALIST TRAINING ON {DEVICE} ---")
    for epoch in range(50):
        model.train()
        epoch_loss = 0
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/50 | Average Loss: {epoch_loss / len(loader):.4f}")
        torch.save(model.state_dict(), "car_specialist_ssd300.pth")