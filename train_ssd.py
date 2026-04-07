import os
import torch
import torch.utils.data
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
import time
from torchvision import transforms as T


# --- 1. CUSTOM TRANSFORMS FOR DETECTION ---
# These classes ensure that when we transform the image,
# the bounding boxes stay aligned with the pixels.

class DetectionCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DetectionHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            # img is still PIL here, so .size is a tuple
            width, _ = img.size
            img = T.RandomHorizontalFlip(p=1.0)(img)
            bbox = target["boxes"]
            # Flip logic: new_xmin = width - old_xmax, new_xmax = width - old_xmin
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return img, target


class DetectionColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.jitter = T.ColorJitter(brightness, contrast, saturation)

    def __call__(self, img, target):
        return self.jitter(img), target


class DetectionToTensor:
    def __call__(self, img, target):
        return T.ToTensor()(img), target


class DetectionRandomErasing:
    def __init__(self, p=0.1):
        self.erase = T.RandomErasing(p=p, scale=(0.02, 0.1))

    def __call__(self, img, target):
        # RandomErasing MUST happen after ToTensor (on a Tensor)
        return self.erase(img), target


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

        boxes = []
        labels = []
        for obj in root.findall('object'):
            label_text = obj.find('name').text.lower().strip()
            if label_text in self.label_map:
                bndbox = obj.find('bndbox')
                xmin = float(bndbox.find('xmin').text)
                ymin = float(bndbox.find('ymin').text)
                xmax = float(bndbox.find('xmax').text)
                ymax = float(bndbox.find('ymax').text)

                if xmax > xmin and ymax > ymin:
                    labels.append(self.label_map[label_text])
                    boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.as_tensor([], dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# --- 3. SETTINGS & PREP ---
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
BATCH_SIZE = 4
EPOCHS = 80


def get_transform(train):
    transforms = []
    if train:
        # 1. PIL-based augmentations FIRST
        transforms.append(DetectionHorizontalFlip(0.5))
        transforms.append(DetectionColorJitter(0.2, 0.2, 0.2))

    # 2. Convert PIL to Tensor (Middle)
    transforms.append(DetectionToTensor())

    if train:
        # 3. Tensor-based augmentations LAST
        transforms.append(DetectionRandomErasing(p=0.1))

    return DetectionCompose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


# Prepare Loaders
train_dataset = XMLDataset(root='dataset/master_train', transforms=get_transform(train=True))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

val_dataset = XMLDataset(root='dataset/master_valid', transforms=get_transform(train=False))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 4. MODEL ---
model = ssd300_vgg16(weights=SSD300_VGG16_Weights.DEFAULT)
num_classes = 5
in_channels = [512, 1024, 512, 256, 256, 256]
num_anchors = [4, 6, 6, 6, 4, 4]
model.head = SSDHead(in_channels, num_anchors, num_classes)
model.to(DEVICE)

# --- 5. TRAINING LOOP ---
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

print("-" * 30)
print(f"TRAINING START | Images: {len(train_dataset)} | Device: {DEVICE}")
print("-" * 30)

start_full_train = time.time()
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    start_epoch = time.time()
    epoch_loss = 0

    for images, targets in train_loader:
        valid_indices = [i for i, t in enumerate(targets) if len(t['boxes']) > 0]
        if not valid_indices: continue

        images = [images[i].to(DEVICE) for i in valid_indices]
        targets = [{k: v.to(DEVICE) for k, v in targets[i].items()} for i in valid_indices]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()

    # --- VALIDATION ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for val_images, val_targets in val_loader:
            v_valid = [i for i, t in enumerate(val_targets) if len(t['boxes']) > 0]
            if not v_valid: continue
            v_imgs = [val_images[i].to(DEVICE) for i in v_valid]
            v_targs = [{k: v.to(DEVICE) for k, v in val_targets[i].items()} for i in v_valid]

            model.train()
            loss_dict = model(v_imgs, v_targs)
            model.eval()
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | LR: {current_lr:.6f} | Time: {(time.time() - start_epoch) / 60:.2f}m")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "malaysian_ssd_BEST.pth")
        print(f"*** NEW BEST MODEL SAVED! ***")

    if (epoch + 1) % 20 == 0:
        torch.save(model.state_dict(), f"checkpoint_epoch_{epoch + 1}.pth")

print(f"FINISHED! Total Time: {(time.time() - start_full_train) / 60:.2f} mins")