"""
Quick Inference & Model Verification Demo for DayViT.
Loads the trained checkpoint and runs evaluation on the held-out test split.

Usage:
    python demo_inference.py
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.dataset import get_train_test_loaders, INV_LABEL_MAP
from src.models import DayViT, count_parameters


def run_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"   DayViT: Cellular Traffic Inference Demo        ")
    print(f"==================================================")
    print(f"Device: {device}")

    checkpoint_path = ROOT_DIR / "checkpoints" / "best_training_tuned_vit.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Initialize Model
    model = DayViT(num_classes=4, embed_dim=48, num_heads=4, num_layers=3).to(device)
    print(f"Total Model Parameters: {count_parameters(model):,}")

    # 2. Load Weights
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Successfully loaded checkpoint: {checkpoint_path.name}\n")

    # 3. Load Test Data
    _, test_loader, norm_stats, _ = get_train_test_loaders(batch_size=32, test_size=0.2, random_state=42)

    # 4. Run Evaluation
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Held-out Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    target_names = [INV_LABEL_MAP[i] for i in range(4)]
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

    # 5. Sample Individual Prediction Demonstration
    print("---------------- Sample Predictions ----------------")
    sample_inputs, sample_labels = next(iter(test_loader))
    sample_inputs = sample_inputs[:5].to(device)
    with torch.no_grad():
        logits = model(sample_inputs)
        probs = F.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

    for i in range(len(sample_inputs)):
        gt_cls = INV_LABEL_MAP[sample_labels[i].item()]
        pred_cls = INV_LABEL_MAP[preds[i].item()]
        conf = probs[i][preds[i]].item() * 100
        status = "CORRECT" if gt_cls == pred_cls else "WRONG"
        print(f"Sample #{i+1}: True=[{gt_cls}] | Pred=[{pred_cls}] ({conf:.1f}% conf) -> {status}")
    print(f"==================================================")


if __name__ == "__main__":
    run_demo()
