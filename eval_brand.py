import os
import torch
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.data import DataLoader
from train_car import XMLDataset, create_ssd512


def fix_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("backbone.") and not k.startswith("backbone.features."):
            new_key = k.replace("backbone.", "backbone.features.")
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def load_eval_model(path, device):
    if not os.path.exists(path): return None, None
    checkpoint = torch.load(path, map_location=device)
    fixed_checkpoint = fix_state_dict(checkpoint)
    for num_classes in [11, 7, 2]:
        try:
            model = create_ssd512(num_classes=num_classes)
            model.load_state_dict(fixed_checkpoint, strict=True)
            return model.to(device).eval(), num_classes
        except:
            continue
    return None, None

def evaluate_brand(model_path, test_folder, label_map, conf_threshold=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, classes = load_eval_model(model_path, device)
    if not model: return

    print(f"✅ Loaded {model_path} ({classes} classes). Starting Brand Evaluation...")
    dataset = XMLDataset([test_folder], label_map, transforms=True)
    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda b: tuple(zip(*b)))

    # class_metrics=True allows per-brand breakdown in your final report
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)

    for images, targets in tqdm(loader):
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
        res = [{"boxes": out["boxes"][out['scores'] > conf_threshold].cpu(),
                "scores": out["scores"][out['scores'] > conf_threshold].cpu(),
                "labels": out["labels"][out['scores'] > conf_threshold].cpu()} for out in outputs]
        tgt = [{"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()} for t in targets]
        metric.update(res, tgt)

    results = metric.compute()
    print(
        f"\nBRAND RESULTS: Precision: {results['map'].item():.4f} | Recall: {results['mar_100'].item():.4f} | mAP@50: {results['map_50'].item():.4f}")


if __name__ == "__main__":
    # Updated to your 1-5 Diagnostic list
    BRAND_MAP = {'honda': 1, 'nissan': 2, 'perodua': 3, 'proton': 4, 'toyota': 5}
    evaluate_brand("ssd512_brands.pth", r"dataset/Car-Brands.voc/valid", BRAND_MAP)