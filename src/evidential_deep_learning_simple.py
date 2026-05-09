#!/usr/bin/env python3
"""
Evidential Deep Learning (EDL) for ADHD Classification.

Uses pre-computed predictions from ResNet18 + BrainIAC as input features.
Learns a Dirichlet parameterization to estimate:
  - Aleatoric (data) uncertainty: inherent noise in the data
  - Epistemic (model) uncertainty: uncertainty due to lack of training data
  - Mutual information: model disagreement

Input: [resnet_prob, brainiac_prob] per subject
Output: Calibrated uncertainty estimates + confidence scores

Reference:
  Amini et al. (2020): "Uncertainty Quantification 360° in Deep Learning"
  https://arxiv.org/abs/2011.01314
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import digamma
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidential Deep Learning for ADHD Classification"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="inference",
        choices=["train", "inference"],
        help="Train or infer",
    )
    parser.add_argument(
        "--brainiac-run",
        type=str,
        default="runs/BrainIAC_best",
        help="BrainIAC predictions dir",
    )
    parser.add_argument(
        "--resnet-run",
        type=str,
        default="runs/ResNet18_best",
        help="ResNet18 predictions dir",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate")
    parser.add_argument("--early-stop", type=int, default=20, help="Early stopping patience")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split for inference",
    )
    parser.add_argument(
        "--runs-root", type=str, default="runs", help="Output directory"
    )
    return parser.parse_args()


class EDLHead(nn.Module):
    """Learns Dirichlet parameters from ensemble predictions."""

    def __init__(self, in_features: int = 2, num_classes: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(x)
        alpha = torch.exp(logits) + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        return alpha, p


def edl_loss(
    alpha: torch.Tensor,
    target: torch.Tensor,
    lam: float = 0.0,
) -> torch.Tensor:
    """EDL loss: KL divergence + regularization."""
    num_classes = alpha.size(1)
    device = alpha.device

    S = alpha.sum(dim=1)
    target_one_hot = F.one_hot(target, num_classes=num_classes).float().to(device)
    digamma_S_np = digamma((S.detach().cpu().numpy()) + 1e-8)
    digamma_alpha_np = digamma((alpha.detach().cpu().numpy()) + 1e-8)
    digamma_diff = torch.tensor(
        digamma_alpha_np - digamma_S_np.reshape(-1, 1),
        dtype=alpha.dtype,
        device=device,
    )

    kl = (target_one_hot * (torch.log(alpha + 1e-10) - torch.log(S.unsqueeze(1) + 1e-10))).sum(
        dim=1
    ) - ((target_one_hot * digamma_diff).sum(dim=1))


    reg = (alpha - 1.0).sum(dim=1) * (1.0 - target_one_hot.sum(dim=1))

    loss = (kl + lam * reg).mean()
    return loss


def load_predictions(run_dir: Path, split: str) -> pd.DataFrame:
    """Load predictions CSV from a run directory."""
    csv_path = run_dir / f"{split}_predictions.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def merge_predictions(
    args: argparse.Namespace, split: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    resnet_df = load_predictions(Path(args.resnet_run), split)
    brainiac_df = load_predictions(Path(args.brainiac_run), split)

    resnet_idx = {str(row["sub_id"]): row for _, row in resnet_df.iterrows()}
    brainiac_idx = {str(row["sub_id"]): row for _, row in brainiac_df.iterrows()}
    common_subs = set(resnet_idx.keys()) & set(brainiac_idx.keys())
    print(f"Common subjects in {split}: {len(common_subs)}")
    if len(common_subs) == 0:
        raise ValueError(f"No common subjects in {split} split!")

    features = []
    labels = []
    sub_ids = []

    for sub_id in sorted(common_subs):
        r_row = resnet_idx[sub_id]
        b_row = brainiac_idx[sub_id]
        r_label = int(r_row["true_label"])
        b_label = int(b_row["true_label"])
        assert r_label == b_label, f"Label mismatch for {sub_id}: {r_label} vs {b_label}"

        r_prob = float(r_row["pred_prob"])
        b_prob = float(b_row["pred_prob"])

        features.append([r_prob, b_prob])
        labels.append(r_label)
        sub_ids.append(str(sub_id))

    return np.array(features), np.array(labels), np.array(sub_ids)


def train_edl(
    model: EDLHead,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> EDLHead:
    """Train EDL head on merged predictions."""
    print("\n[TRAIN] Evidential Deep Learning Head")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False
    )

    best_auc = -1
    best_state = None
    no_improve = 0

    for epoch in range(args.epochs):
        model.train()
        alpha, p = model(X_train)
        loss = edl_loss(alpha, y_train, lam=0.01)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            alpha_val, p_val = model(X_val)
            val_probs = p_val[:, 1].cpu().numpy()

        val_auc = (
            roc_auc_score(y_val.cpu().numpy(), val_probs)
            if len(np.unique(y_val.cpu().numpy())) > 1
            else 0.5
        )

        scheduler.step(val_auc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{args.epochs} | Train Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= args.early_stop:
            print(f"Early stop at epoch {epoch + 1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, output_dir / "best_edl_head.pth")
        print(f"Best model saved. Best Val AUC: {best_auc:.4f}")

    return model


def infer_edl(
    model: EDLHead, X: torch.Tensor, y: np.ndarray, sub_ids: np.ndarray, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        alpha, p = model(X)

    alphas = alpha.cpu().numpy()
    probs = p.cpu().numpy()
    mean_prob = probs[:, 1]


    S = alphas.sum(axis=1)
    p0, p1 = probs[:, 0], probs[:, 1]
    aleatoric = np.sqrt((p0 * p1) / (S + 1))


    epistemic = np.sqrt((p1 * (1 - p1)) / (S + 1))

    # Mutual information (expected data entropy - expected model entropy)
    mutual_info = np.zeros(len(alphas))
    for i in range(len(alphas)):
        alpha_i = alphas[i]
        S_i = S[i]
        p_i = probs[i]

        digamma_alpha = digamma(alpha_i + 1e-8)
        digamma_S = digamma(S_i + 1e-8)
        entropy_exp = -np.sum(p_i * (digamma_alpha - digamma_S))
        var_exp = np.sum(p_i * (1 - p_i) / (S_i + 1))

        mutual_info[i] = entropy_exp - var_exp

    return y, sub_ids, mean_prob, aleatoric, epistemic, mutual_info


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.runs_root) / f"edl_brainiac_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("EVIDENTIAL DEEP LEARNING (EDL) - ADHD Classification")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}\n")

    if args.mode == "train":
        # Load all splits
        X_train, y_train, sub_ids_train = merge_predictions(args, "train")
        X_val, y_val, sub_ids_val = merge_predictions(args, "val")
        X_test, y_test, sub_ids_test = merge_predictions(args, "test")

        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

        print(
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )

        # Train
        model = EDLHead(in_features=2, num_classes=2, hidden=64).to(device)
        model = train_edl(
            model, X_train_t, y_train_t, X_val_t, y_val_t, args, device, output_dir
        )

        # Evaluate on test
        y_true, sub_ids, mean_prob, aleatoric, epistemic, mutual_info = infer_edl(
            model, X_test_t, y_test, sub_ids_test, device
        )

    else:
        # Inference mode
        X, y, sub_ids = merge_predictions(args, args.split)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)

        print(f"Evaluating split '{args.split}': {len(X)} subjects\n")

        # Load pre-trained model if exists
        model = EDLHead(in_features=2, num_classes=2, hidden=64).to(device)

        # Try to find best checkpoint from most recent training run
        import glob
        matching = sorted(glob.glob(str(Path(args.runs_root) / "edl_brainiac_*" / "best_edl_head.pth")))
        if matching:
            latest_ckpt = matching[-1]
            try:
                model.load_state_dict(torch.load(latest_ckpt, map_location=device))
                print(f"Loaded checkpoint: {latest_ckpt}\n")
            except Exception as e:
                print(f"[WARNING] Could not load checkpoint: {e}\n")

        y_true, sub_ids, mean_prob, aleatoric, epistemic, mutual_info = infer_edl(
            model, X_t, y, sub_ids, device
        )

    # Evaluate metrics
    pred_label = (mean_prob > 0.5).astype(int)

    auc = (
        roc_auc_score(y_true, mean_prob)
        if len(np.unique(y_true)) > 1
        else float("nan")
    )
    acc = accuracy_score(y_true, pred_label)
    f1 = f1_score(y_true, pred_label, zero_division=0)
    cm = confusion_matrix(y_true, pred_label, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else [cm[0, 0], 0, 0, cm[1, 1]]
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n--- Results ({args.split if args.mode == 'inference' else 'test'} set) ---")
    print(f"AUC:         {auc:.4f}")
    print(f"Accuracy:    {acc:.4f}")
    print(f"Sensitivity: {sens:.4f}")
    print(f"Specificity: {spec:.4f}")
    print(f"F1:          {f1:.4f}")

    print(f"\n--- Uncertainty Estimates ---")
    print(f"Mean aleatoric std:  {aleatoric.mean():.4f}")
    print(f"Mean epistemic std:  {epistemic.mean():.4f}")
    print(f"Mean mutual info:    {mutual_info.mean():.4f}")

    # Save results
    results_df = pd.DataFrame(
        {
            "sub_id": sub_ids,
            "true_label": y_true,
            "mean_prob": mean_prob,
            "pred_label": pred_label,
            "aleatoric_std": aleatoric,
            "epistemic_std": epistemic,
            "mutual_info": mutual_info,
        }
    )
    results_df.to_csv(
        output_dir / f"{args.split if args.mode == 'inference' else 'test'}_edl_predictions.csv",
        index=False,
    )

    metrics_dict = {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "Sensitivity": float(sens),
        "Specificity": float(spec),
        "F1": float(f1),
        "Aleatoric_Mean": float(aleatoric.mean()),
        "Epistemic_Mean": float(epistemic.mean()),
        "MutualInfo_Mean": float(mutual_info.mean()),
    }
    pd.DataFrame([metrics_dict]).to_csv(output_dir / "metrics.csv", index=False)

    summary = {
        "timestamp": timestamp,
        "device": str(device),
        "mode": args.mode,
        "split": args.split if args.mode == "inference" else "test",
        "n_subjects": int(len(results_df)),
        "metrics": metrics_dict,
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
