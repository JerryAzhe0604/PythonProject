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


# ==============================
# Detection Transforms
# ==============================

class DetectionCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class DetectionResize:
    def __init__(self, size=(512, 512)):
        self.size = size

    def __call__(self, img, target):
        w, h = img.size
        img = T.Resize(self.size)(img)

        scale_x = self.size[0] / w
        scale_y = self.size[1] / h

        boxes = target["boxes"]
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        target["boxes"] = boxes

        return img, target


class DetectionHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if torch.rand(1) < self.p:
            width, _ = img.size
            img = T.RandomHorizontalFlip(p=1.0)(img)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
        return img, target


class DetectionColorJitter:
    def __init__(self):
        self.jitter = T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.05
        )

    def __call__(self, img, target):
        img = self.jitter(img)
        return img, target


class DetectionNormalize:
    def __init__(self):
        self.norm = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def __call__(self, img, target):
        img = self.norm(img)
        return img, target


# ==============================
# Dataset
# ==============================

class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = []

        for r, d, f in os.walk(root):
            for file in f:
                if file.endswith('.jpg'):
                    self.imgs.append(os.path.join(r, file))

        self.label_map = {
            'plate': 1,
            'car': 2,
            'logo_perodua': 3,
            'logo_proton': 4,
            'logo_honda': 5,
            'logo_toyota': 6,
            'logo_mercedes': 7,
            'logo_bmw': 8,
            'logo_nissan': 9,
            'logo_others': 10
        }

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        xml_path = img_path.replace('.jpg', '.xml')

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.autocontrast(img)

        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text.lower().strip()

            if name in self.label_map:
                bnd = obj.find('bndbox')

                xmin = float(bnd.find('xmin').text)
                ymin = float(bnd.find('ymin').text)
                xmax = float(bnd.find('xmax').text)
                ymax = float(bnd.find('ymax').text)

                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(self.label_map[name])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64)
        }

        img = T.ToTensor()(img)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


# ==============================
# Model
# ==============================

def create_ssd512(num_classes):

    backbone = vgg16(
        weights=VGG16_Weights.IMAGENET1K_V1
    ).features

    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )

    head = SSDHead(
        [512, 1024, 512, 256, 256, 256, 256],
        [4, 6, 6, 6, 6, 4, 4],
        num_classes
    )

    model = SSD(
        backbone,
        anchor_gen,
        (512, 512),
        num_classes,
        head=head
    )

    return model


# ==============================
# Training Setup
# ==============================

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = create_ssd512(num_classes=11).to(DEVICE)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)

scaler = torch.amp.GradScaler('cuda')


# ==============================
# Dataset
# ==============================

dataset = XMLDataset(
    'dataset/sorted_train',
    transforms=DetectionCompose([
        DetectionResize(),
        DetectionHorizontalFlip(),
        DetectionColorJitter(),
        DetectionNormalize()
    ])
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=4,
    collate_fn=lambda b: tuple(zip(*b))
)


# ==============================
# Training Loop
# ==============================

warmup_epochs = 3

for epoch in range(50):

    model.train()
    total_loss = 0

    # Warmup
    if epoch < warmup_epochs:
        lr = 0.001 * (epoch + 1) / warmup_epochs
        for g in optimizer.param_groups:
            g['lr'] = lr

    for images, targets in loader:

        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda'):

            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())

        if not torch.isfinite(loss):
            continue

        optimizer.zero_grad()

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            2.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    scheduler.step()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    torch.save(
        model.state_dict(),
        "malaysian_ssd512_improved.pth"
    )

print("Training Complete")