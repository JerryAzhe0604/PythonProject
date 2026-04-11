import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from train_ssd512 import XMLDataset, create_ssd512, DetectionNormalize  # Import your classes
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- SETTINGS ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "malaysian_ssd512_RESCUE.pth"
VALID_PATH = "dataset/master_valid"  # Change to your validation folder


def evaluate():
    # 1. Load Model
    model = create_ssd512(num_classes=11).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load Validation Data
    dataset = XMLDataset(VALID_PATH, transforms=DetectionNormalize())
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=lambda b: tuple(zip(*b)))

    # 3. Setup Metrics
    metric = MeanAveragePrecision(iou_type="bbox")

    print(f"📊 Running Evaluation on {len(dataset)} images...")

    with torch.no_grad():
        for images, targets in tqdm(loader):
            images = [img.to(DEVICE) for img in images]

            # Get Predictions
            outputs = model(images)

            # Format predictions and targets for torchmetrics
            preds = []
            for out in outputs:
                preds.append({
                    "boxes": out["boxes"].cpu(),
                    "scores": out["scores"].cpu(),
                    "labels": out["labels"].cpu()
                })

            target_list = []
            for t in targets:
                target_list.append({
                    "boxes": t["boxes"].cpu(),
                    "labels": t["labels"].cpu()
                })

            metric.update(preds, target_list)

    # 4. Compute Final Values
    results = metric.compute()

    # --- PRINT YOUR FYP VALUES ---
    print("\n" + "=" * 30)
    print("🚀 FINAL EVALUATION RESULTS")
    print("=" * 30)
    print(f"📍 mAP @ 50: {results['map_50'].item():.4f}")
    print(f"📍 mAP @ [50:95]: {results['map'].item():.4f}")

    # Precision and Recall are calculated at specific IOU thresholds
    # These are usually extracted from the results dictionary
    print(f"📍 Average Recall: {results['mar_100'].item():.4f}")

    # Calculation of F1-Score (Manual)
    prec = results['map_50'].item()
    rec = results['mar_100'].item()
    if (prec + rec) > 0:
        f1 = 2 * (prec * rec) / (prec + rec)
        print(f"📍 F1-Score: {f1:.4f}")
    else:
        print("📍 F1-Score: 0.0000")
    print("=" * 30)


if __name__ == "__main__":
    evaluate()