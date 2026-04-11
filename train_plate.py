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


class XMLDataset(torch.utils.data.Dataset):
    def __init__(self, roots, label_map, transforms=None):
        self.roots = roots if isinstance(roots, list) else [roots]
        self.transforms = transforms
        self.label_map = label_map
        self.imgs = []
        for root in self.roots:
            for r, d, f in os.walk(root):
                for file in f:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.imgs.append(os.path.join(r, file))

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
        if self.transforms:
            img = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, target

    def __len__(self):
        return len(self.imgs)


def create_ssd512(num_classes):
    backbone = nn.Sequential(*list(vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features)[:30])
    anchor_gen = DefaultBoxGenerator([[2], [2, 3], [2, 3], [2, 3], [2], [2], [2]],
                                     scales=[0.04, 0.1, 0.26, 0.42, 0.58, 0.74, 0.9, 1.06, 1.20],
                                     steps=[8, 16, 32, 64, 128, 256, 512])
    head = SSDHead([512, 1024, 512, 256, 256, 256, 256], [4, 6, 6, 6, 4, 4, 4], num_classes)
    return SSD(backbone, anchor_gen, (512, 512), num_classes, head=head)


if __name__ == "__main__":
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_ssd512(num_classes=11).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler()

    # FOCUS: License Plate Dataset
    train_folders = [r"dataset/License-Plate.v1i.voc/train"]
    l_map = {'plate': 1, 'license-plate': 1}

    dataset = XMLDataset(train_folders, l_map, transforms=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=lambda b: tuple(zip(*b)))

    print(f"--- TRAINING PLATE SPECIALIST: {len(dataset)} images ---")
    for epoch in range(50):
        model.train();
        epoch_loss = 0
        for images, targets in loader:
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                loss = sum(l for l in loss_dict.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward();
            scaler.step(optimizer);
            scaler.update()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {epoch_loss / len(loader):.4f}")
        torch.save(model.state_dict(), "plate_specialist.pth")