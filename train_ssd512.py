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
# 1. FIXED TRANSFORMS (No more double-resizing)
# ==========================================
class DetectionCompose:
    def __init__(self, transforms): self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms: img, target = t(img, target)
        return img, target


class DetectionHorizontalFlip:
    def __init__(self, p=0.5): self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            width, _ = img.size
            img = T.RandomHorizontalFlip(p=1.0)(img)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return img, target


class DetectionNormalize:
    def __init__(self):
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __call__(self, img, target):
        img = self.norm(img)
        return img, target


# ==========================================
# 2. DATASET (Synced Label Map)
# ==========================================
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [os.path.join(r, file) for r, d, f in os.walk(root) for file in f if file.endswith('.jpg')]
        self.label_map = {
            'plate': 1, 'car': 2, 'logo_perodua': 3, 'logo_proton': 4,
            'logo_honda': 5, 'logo_toyota': 6, 'logo_mercedes': 7,
            'logo_bmw': 8, 'logo_nissan': 9, 'logo_others': 10
        }

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        xml_path = img_path.replace('.jpg', '.xml')
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.autocontrast(img)

        tree = ET.parse(xml_path);
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

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        # Internal Resize to 512 for SSD consistency
        w, h = img.size
        img = img.resize((512, 512))
        target["boxes"][:, [0, 2]] *= (512.0 / w)
        target["boxes"][:, [1, 3]] *= (512.0 / h)

        img = T.ToTensor()(img)
        if self.transforms: img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# ==========================================
# 3. MODEL (The "Sync-Fix" Architecture)
# ==========================================
def create_ssd512(num_classes):
    # Slice VGG16 features at layer 30 to get 512 channels
    backbone = nn.Sequential(*list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)[:30])

    # Anchors: Synced with the [4, 6, 6, 6, 4, 4, 4] layout
    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )

    # Head: Synced with your current weights to prevent size mismatch
    head = SSDHead(
        [512, 1024, 512, 256, 256, 256, 256],
        [4, 6, 6, 6, 4, 4, 4],  # <--- CRITICAL FIX: The 44-param layer
        num_classes
    )

    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


# ==========================================
# 4. TRAINING LOOP
# ==========================================
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_ssd512(num_classes=11).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
scaler = torch.amp.GradScaler('cuda')

dataset = XMLDataset('dataset/sorted_train',
                     transforms=DetectionCompose([DetectionHorizontalFlip(), DetectionNormalize()]))
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))

print(f"--- TRAINING START (Device: {DEVICE}) ---")
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

        if not torch.isfinite(loss): continue

        optimizer.zero_grad();
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer);
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer);
        scaler.update()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "malaysian_ssd512_RESCUE.pth")

print("Training Finished. Use 'malaysian_ssd512_RESCUE.pth' in your app.")