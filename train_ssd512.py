import os
import torch
import torch.nn as nn
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision
from torchvision.models.detection.ssd import SSD, SSDHead, DefaultBoxGenerator
from torchvision.models.vgg import vgg16, VGG16_Weights
import time
from torchvision import transforms as T


# --- 1. CUSTOM TRANSFORMS ---
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


class DetectionColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.jitter = T.ColorJitter(brightness, contrast, saturation)

    def __call__(self, img, target): return self.jitter(img), target


class DetectionToTensor:
    def __call__(self, img, target): return T.ToTensor()(img), target


class DetectionRandomErasing:
    def __init__(self, p=0.1): self.erase = T.RandomErasing(p=p, scale=(0.02, 0.1))

    def __call__(self, img, target): return self.erase(img), target


# --- 2. DATASET CLASS ---
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = [f for f in os.listdir(root) if f.endswith('.jpg')]
        self.label_map = {'plate': 1, 'car': 2, 'logo': 3, 'model_badge': 4}

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        xml_path = os.path.join(self.root, self.imgs[idx].replace('.jpg', '.xml'))
        img = Image.open(img_path).convert("RGB")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        boxes, labels = [], []
        for obj in root.findall('object'):
            label_text = obj.find('name').text.lower().strip()
            if label_text in self.label_map:
                bndbox = obj.find('bndbox')
                xmin, ymin = float(bndbox.find('xmin').text), float(bndbox.find('ymin').text)
                xmax, ymax = float(bndbox.find('xmax').text), float(bndbox.find('ymax').text)
                if xmax > xmin and ymax > ymin:
                    labels.append(self.label_map[label_text])
                    boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes if boxes else torch.zeros((0, 4)), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        if self.transforms is not None: img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# --- 3. SETTINGS ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 1  # Keep 1 for 4GB VRAM
ACCUMULATION_STEPS = 4
EPOCHS = 50


def get_transform(train):
    transforms = []
    if train:
        transforms.append(DetectionHorizontalFlip(0.5))
        transforms.append(DetectionColorJitter(0.2, 0.2, 0.2))
    transforms.append(DetectionToTensor())
    if train: transforms.append(DetectionRandomErasing(p=0.1))
    return DetectionCompose(transforms)


def collate_fn(batch): return tuple(zip(*batch))


# Paths remain exactly as they were in your folder structure
train_dataset = XMLDataset(root='dataset/master_train', transforms=get_transform(train=True))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

val_dataset = XMLDataset(root='dataset/master_valid', transforms=get_transform(train=False))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


# --- 4. SSD512 MODEL CONSTRUCTION ---
def create_ssd512(num_classes):
    backbone_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    backbone = nn.Sequential(*list(backbone_vgg)[:30])

    anchor_generator = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )

    in_channels = [512, 1024, 512, 256, 256, 256, 256]
    num_anchors = [4, 6, 6, 6, 6, 4, 4]
    head = SSDHead(in_channels, num_anchors, num_classes)

    # Assembly with correct num_classes argument
    return SSD(backbone=backbone, anchor_generator=anchor_generator, size=(512, 512), num_classes=num_classes,
               head=head)


model = create_ssd512(num_classes=5).to(DEVICE)

# --- 5. SAFETY-OPTIMIZED TRAINING LOOP ---
# Reduced Learning Rate (0.0001) for stability
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
scaler = torch.amp.GradScaler('cuda')

print("-" * 40)
print(f"GTX 1650 SAFETY RESTART | SSD512 | Device: {DEVICE}")
print(f"LR: 0.0001 | Training on master_train...")
print("-" * 40)

start_full_train = time.time()
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    start_epoch = time.time()
    epoch_loss = 0
    optimizer.zero_grad()

    for i, (images, targets) in enumerate(train_loader):
        valid_indices = [idx for idx, t in enumerate(targets) if len(t['boxes']) > 0]
        if not valid_indices: continue

        images = [images[idx].to(DEVICE) for idx in valid_indices]
        targets = [{k: v.to(DEVICE) for k, v in targets[idx].items()} for idx in valid_indices]

        # Stable AMP syntax
        with torch.amp.autocast('cuda'):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses = losses / ACCUMULATION_STEPS

            # NaN/Inf Check
        if not torch.isfinite(losses):
            continue

        scaler.scale(losses).backward()

        if (i + 1) % ACCUMULATION_STEPS == 0:
            # Gradient Clipping (Safety Cap)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_loss += losses.item() * ACCUMULATION_STEPS

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            v_valid = [idx for idx, t in enumerate(val_targets) if len(t['boxes']) > 0]
            if not v_valid: continue
            v_imgs = [val_images[idx].to(DEVICE) for idx in v_valid]
            v_targs = [{k: v.to(DEVICE) for k, v in val_targets[idx].items()} for idx in v_valid]

            with torch.amp.autocast('cuda'):
                model.train()
                v_loss_dict = model(v_imgs, v_targs)
                model.eval()
                v_losses = sum(loss for loss in v_loss_dict.values())
            val_loss += v_losses.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | Time: {(time.time() - start_epoch) / 60:.2f}m")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "malaysian_ssd512_allinone.pth")
        print(f"*** BEST MODEL SAVED! ***")

print(f"FINISHED! Total Time: {(time.time() - start_full_train) / 60:.2f} mins")