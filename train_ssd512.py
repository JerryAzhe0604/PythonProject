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
# 1. ROBUST TRANSFORMS
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
            if hasattr(img, 'shape'):
                width = img.shape[-1]
            else:
                width, _ = img.size
            img = torch.flip(img, dims=[-1]) if hasattr(img, 'shape') else T.RandomHorizontalFlip(p=1.0)(img)
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
# 2. MULTI-DATASET LOADER (With Debugger)
# ==========================================
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, roots, transforms=None):
        self.roots = roots if isinstance(roots, list) else [roots]
        self.transforms = transforms
        self.imgs = []

        print(f"\n--- PATH DEBUGGER ---")
        print(f"Working Directory: {os.getcwd()}")

        for root in self.roots:
            full_path = os.path.abspath(root)
            print(f"Scanning: {full_path}")
            if not os.path.exists(root):
                print(f"!! ERROR: Folder not found: {root}")
                continue

            for r, d, f in os.walk(root):
                for file in f:
                    # Catch .jpg, .JPG, .jpeg, .png
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.imgs.append(os.path.join(r, file))

        print(f"Total Images Found: {len(self.imgs)}")
        print(f"----------------------\n")

        self.label_map = {
            'license-plate': 1, 'plate': 1, 'car': 2, 'vehicle': 2,
            'perodua': 3, 'proton': 4, 'honda': 5, 'toyota': 6,
            'mercedes': 7, 'bmw': 8, 'nissan': 9, 'others': 10
        }

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        xml_path = img_path.rsplit('.', 1)[0] + '.xml'
        img = Image.open(img_path).convert("RGB")
        img = ImageOps.autocontrast(img)

        try:
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
        except:
            return self.__getitem__((idx + 1) % len(self))

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

    def __len__(self):
        return len(self.imgs)


# ==========================================
# 3. SYNCED ARCHITECTURE
# ==========================================
def create_ssd512(num_classes):
    backbone = nn.Sequential(*list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)[:30])
    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06, 1.20],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )
    head = SSDHead([512, 1024, 512, 256, 256, 256, 256], [4, 6, 6, 6, 4, 4, 4], num_classes)
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


# ==========================================
# 4. TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_ssd512(num_classes=11).to(DEVICE)

    # AdamW at 0.001 is best for fast learning on good hardware
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = torch.cuda.amp.GradScaler()

    # --- SET PATHS HERE ---
    # Using 'r' for Windows paths to avoid backslash issues
    train_folders = [
        r"dataset/Car Brands.v3-carbrands.voc/train",
        r"dataset/Car Models.v2-carobject.voc/train"
    ]

    dataset = XMLDataset(train_folders,
                         transforms=DetectionCompose([DetectionHorizontalFlip(), DetectionNormalize()]))

    # batch_size=8 is safe for most 8GB+ GPUs
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                         collate_fn=lambda b: tuple(zip(*b)), num_workers=0)

    if len(dataset) == 0:
        print("!! TERMINATING: No images found. Check your train_folders paths.")
    else:
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
                scaler.step(optimizer);
                scaler.update()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(loader):.4f}")
            torch.save(model.state_dict(), "malaysian_ssd512_RESCUE.pth")