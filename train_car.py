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
# 1. MULTI-DATASET LOADER (Specialist Edition)
# ==========================================
class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, roots, label_map, transforms=None):
        self.roots = roots if isinstance(roots, list) else [roots]
        self.transforms = transforms
        self.label_map = label_map
        self.imgs = []

        print(f"\n--- PATH DEBUGGER ---")
        for root in self.roots:
            full_path = os.path.abspath(root)
            if not os.path.exists(root):
                print(f"!! WARNING: Folder not found: {full_path}")
                continue
            print(f"Scanning folder: {full_path}")
            for r, d, f in os.walk(root):
                for file in f:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.imgs.append(os.path.join(r, file))

        print(f"Total Images Found: {len(self.imgs)}")
        print(f"----------------------\n")

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        xml_path = img_path.rsplit('.', 1)[0] + '.xml'

        # --- DEBUG PRINT FOR EACH IMAGE ---
        print(f"DEBUG: Processing {os.path.basename(img_path)}")

        if not os.path.exists(xml_path):
            print(f"  !! ERROR: XML missing for {os.path.basename(img_path)}")
            return self.__getitem__((idx + 1) % len(self))

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.autocontrast(img)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            boxes, labels = [], []

            for obj in root.findall('object'):
                name_tag = obj.find('name')
                if name_tag is not None:
                    name = name_tag.text.lower().strip()

                    if name in self.label_map:
                        bnd = obj.find('bndbox')
                        xmin = float(bnd.find('xmin').text)
                        ymin = float(bnd.find('ymin').text)
                        xmax = float(bnd.find('xmax').text)
                        ymax = float(bnd.find('ymax').text)

                        if xmax > xmin and ymax > ymin:
                            boxes.append([xmin, ymin, xmax, ymax])
                            labels.append(self.label_map[name])
                            print(f"  -> Match Found: '{name}' (ID: {self.label_map[name]})")
                        else:
                            print(f"  -> Skipping: Invalid box size for '{name}'")
                    else:
                        print(f"  -> Skipping: '{name}' is not in your l_map")

            if len(boxes) == 0:
                print(f"  !! WARNING: 0 objects found in {os.path.basename(xml_path)}")

        except Exception as e:
            print(f"!! XML READ ERROR: {e}")
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
        if self.transforms:
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


# ==========================================
# 2. ARCHITECTURE (Must match app.py)
# ==========================================
def create_ssd512(num_classes):
    vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features

    class BackboneWrapper(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.features = nn.Sequential(*list(features)[:30])

        def forward(self, x): return self.features(x)

    backbone = BackboneWrapper(vgg)
    anchor_gen = DefaultBoxGenerator(
        [[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
        scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06, 1.20],
        steps=[8, 16, 32, 64, 128, 256, 512]
    )
    head = SSDHead([512, 1024, 512, 256, 256, 256, 256], [4, 6, 6, 6, 4, 4, 4], num_classes)
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


# ==========================================
# 3. TRAINING EXECUTION
# ==========================================
if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_ssd512(num_classes=11).to(DEVICE)

    # Using SGD with a lower learning rate for stability (Prevents Box Soup)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    scaler = torch.cuda.amp.GradScaler()

    # --- PATHS (Update if needed) ---
    train_folders = [r"Car Models.v2-carobject.voc/train"]

    # --- UNIVERSAL CAR MAP (Catching all possible names) ---
    l_map = {
        'cars': 2,  # Matches <name>Cars</name> after .lower()
        'car': 2,  # Safety for other datasets
        'vehicle': 2,  # Safety for other datasets
        'license-plate': 1,
        'plate': 1
    }

    dataset = XMLDataset(train_folders, l_map, transforms=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True,
                                         collate_fn=lambda b: tuple(zip(*b)), num_workers=0)

    if len(dataset) == 0:
        print("!! TERMINATING: No images found. Check folder paths.")
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
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(loader):.4f}")
            torch.save(model.state_dict(), "car_specialist.pth")