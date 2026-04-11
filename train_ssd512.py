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
# 1. SMART TRANSFORMS (Prevents Unpack Error)
# ==========================================
class DetectionCompose:
    def __init__(self, transforms): self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms: img, target = t(img, target)
        return img, target


class DetectionHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            # Check if it's a Tensor or PIL Image to get width correctly
            if hasattr(img, 'shape'):  # It's a Tensor [C, H, W]
                width = img.shape[-1]
                img = torch.flip(img, dims=[-1])
            else:  # It's a PIL Image
                width, _ = img.size
                img = T.RandomHorizontalFlip(p=1.0)(img)

            # Flip the bounding boxes
            bbox = target["boxes"]
            # x_new = width - x_old
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
# 2. DATASET
# ==========================================
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # This list finds all your images
        self.imgs = [os.path.join(r, file) for r, d, f in os.walk(root) for file in f if file.endswith('.jpg')]

        self.label_map = {
            'plate': 1, 'car': 2, 'logo_perodua': 3, 'logo_proton': 4,
            'logo_honda': 5, 'logo_toyota': 6, 'logo_mercedes': 7,
            'logo_bmw': 8, 'logo_nissan': 9, 'logo_others': 10
        }

    # 1. This tells the AI HOW to get one image
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

        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        w, h = img.size
        img = img.resize((512, 512))
        if boxes.shape[0] > 0:
            target["boxes"][:, [0, 2]] *= (512.0 / w)
            target["boxes"][:, [1, 3]] *= (512.0 / h)

        img = T.ToTensor()(img)
        if self.transforms: img, target = self.transforms(img, target)
        return img, target

    # 2. THIS IS THE PART YOU ARE MISSING
    # It tells the DataLoader how many images are in the folder
    def __len__(self):
        return len(self.imgs)


# ==========================================
# 3. ARCHITECTURE SYNC (Matches your app.py)
# ==========================================
def create_ssd512(num_classes):
    backbone = nn.Sequential(*list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)[:30])
    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )
    head = SSDHead(
        [512, 1024, 512, 256, 256, 256, 256],
        [4, 6, 6, 6, 4, 4, 4],  # This fixes the 44-param mismatch for your app
        num_classes
    )
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


# ==========================================
# 4. TRAINING LOOP
# ==========================================
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = create_ssd512(num_classes=11).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

# Version-safe settings for Anaconda environments
scaler = torch.cuda.amp.GradScaler()

dataset = XMLDataset('dataset/sorted_train',
                     transforms=DetectionCompose([DetectionHorizontalFlip(), DetectionNormalize()]))

# num_workers=0 is safer for Windows/Anaconda
loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
                                     collate_fn=lambda b: tuple(zip(*b)), num_workers=0)

print(f"--- TRAINING START on {DEVICE} ---")
for epoch in range(50):
    model.train()
    epoch_loss = 0
    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

        if not torch.isfinite(loss): continue

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(loader):.4f}")
    torch.save(model.state_dict(), "malaysian_ssd512_RESCUE.pth")

print("Training Done!")