#!/usr/bin/env python3
"""
MC Dropout Bayesian Inference for ADHD ResNet18

Purpose
- Approximate Bayesian Neural Network uncertainty using Monte Carlo (MC) Dropout.
- Reuse an existing trained ResNet18 classifier without retraining.
- Produce per-subject predictive mean, uncertainty, and uncertainty-aware metrics.

Notes
- Best results require a checkpoint trained with dropout layers in the classifier head.
- If no fine-tuned checkpoint is found, this script can fall back to pretrained
  MedicalNet weights, but those outputs are not task-calibrated for ADHD labels.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from monai.data import DataLoader, Dataset
from monai.networks.nets import ResNet
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, NormalizeIntensityd, Resized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MC Dropout BNN inference on ResNet18")
    parser.add_argument("--bids-root", type=str, default="ADHD_BIDS", help="Path to BIDS root directory")
    parser.add_argument("--runs-root", type=str, default="runs", help="Root runs directory")
    parser.add_argument(
        "--source",
        type=str,
        default="split",
        choices=["split", "participants"],
        help="Use labels from split CSVs or from participants.tsv",
    )
    parser.add_argument(
        "--resnet-run-dir",
        type=str,
        default="runs/ResNet18_best",
        help="Directory containing split CSVs (train/val/test predictions)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to fine-tuned ResNet18 checkpoint. If empty, auto-tries run dir, then pretrained fallback.",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="If no checkpoint can be loaded, continue with randomly initialized model (debug only)",
    )
    parser.add_argument(
        "--pretrained-fallback",
        type=str,
        default="pretrained_models/resnet_18.pth",
        help="Fallback checkpoint path if fine-tuned checkpoint not found",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--target-size", nargs=3, type=int, default=[121, 128, 121], help="3D resize target")
    parser.add_argument("--dropout-rate", type=float, default=0.5, help="Dropout rate for FC head")
    parser.add_argument("--mc-samples", type=int, default=30, help="Number of stochastic forward passes")
    parser.add_argument("--batch-size", type=int, default=2, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        default=0.08,
        help="Std-dev threshold for marking uncertain predictions",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional cap on number of samples from split for quick checks (0 = all)",
    )
    return parser.parse_args()


def build_resnet18(dropout_rate: float, device: torch.device) -> nn.Module:
    model = ResNet(
        block=ResNetBlock,
        layers=[2, 2, 2, 2],
        block_inplanes=[64, 128, 256, 512],
        spatial_dims=3,
        n_input_channels=1,
        num_classes=2,
    ).to(device)
    if dropout_rate > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout_rate), model.fc)
    return model


def _clean_state_dict(raw_state: dict, model: nn.Module) -> dict:
    if "state_dict" in raw_state and isinstance(raw_state["state_dict"], dict):
        raw_state = raw_state["state_dict"]
    clean = {}
    model_keys = model.state_dict().keys()
    for key, value in raw_state.items():
        new_key = key.replace("module.", "")
        if new_key in model_keys:
            clean[new_key] = value
    return clean


def load_checkpoint_with_fallback(model: nn.Module, args: argparse.Namespace) -> Tuple[str, List[str]]:
    logs: List[str] = []

    candidates: List[Path] = []
    if args.checkpoint:
        candidates.append(Path(args.checkpoint))

    auto_best = Path(args.resnet_run_dir) / "best_ResNet18.pth"
    auto_alt = Path(args.resnet_run_dir) / "model.pth"
    candidates.extend([auto_best, auto_alt, Path(args.pretrained_fallback)])

    loaded_path = ""
    for candidate in candidates:
        if candidate.exists():
            try:
                ckpt = torch.load(str(candidate), map_location="cpu")
                clean = _clean_state_dict(ckpt, model)
                missing, unexpected = model.load_state_dict(clean, strict=False)
                loaded_path = str(candidate)
                logs.append(f"Loaded checkpoint: {candidate}")
                if missing:
                    logs.append(f"Missing keys (strict=False): {len(missing)}")
                if unexpected:
                    logs.append(f"Unexpected keys (strict=False): {len(unexpected)}")
                break
            except Exception as exc:
                logs.append(f"Failed to load {candidate}: {type(exc).__name__}: {exc}")

    if not loaded_path:
        if args.allow_random_init:
            logs.append("No valid checkpoint loaded; proceeding with random initialization (--allow-random-init).")
            return "random_init", logs
        raise FileNotFoundError(
            "No valid checkpoint could be loaded. Provide --checkpoint with a real .pth file "
            "(not a Git LFS pointer) or use --allow-random-init for debugging."
        )

    return loaded_path, logs


def enable_dropout_at_inference(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def build_image_index(bids_root: Path) -> dict:
    """
    Build sub_id -> image path index from all .nii/.nii.gz under bids_root.
    Supports both canonical BIDS paths and site-based folder layouts.
    """
    pattern = re.compile(r"sub-(\d+)")
    index = {}

    nii_files = list(bids_root.rglob("*.nii")) + list(bids_root.rglob("*.nii.gz"))
    def _is_probably_loadable(path: Path) -> bool:
        s = str(path)
        if s.endswith(".nii.gz"):
            try:
                with open(path, "rb") as f:
                    magic = f.read(2)
                return magic == b"\x1f\x8b"
            except Exception:
                return False
        return True

    for fp in nii_files:
        if not _is_probably_loadable(fp):
            continue

        path_str = str(fp)
        match = pattern.search(path_str)
        if not match:
            continue
        sub_id = match.group(1)

        if sub_id not in index:
            index[sub_id] = fp
            continue

        current = str(index[sub_id])
        candidate = path_str

        current_score = int("biascorr_brain" in current) + int("_T1w" in current) - int("/anat/" in current)
        candidate_score = int("biascorr_brain" in candidate) + int("_T1w" in candidate) - int("/anat/" in candidate)
        if candidate_score > current_score:
            index[sub_id] = fp

    return index


def make_split_df(args: argparse.Namespace, image_index: dict) -> pd.DataFrame:
    split_csv = Path(args.resnet_run_dir) / f"{args.split}_predictions.csv"
    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    if "sub_id" not in df.columns or "true_label" not in df.columns:
        raise ValueError(f"Expected columns ['sub_id', 'true_label'] in {split_csv}")

    df["sub_id"] = df["sub_id"].astype(str)

    paths = []
    keep = []
    for _, row in df.iterrows():
        sub_id = row["sub_id"]
        img_path = Path(args.bids_root) / f"sub-{sub_id}" / "anat" / f"{sub_id}_T1w.nii.gz"
        if sub_id in image_index:
            keep.append(True)
            paths.append(str(image_index[sub_id]))
        elif img_path.exists():
            keep.append(True)
            paths.append(str(img_path))
        else:
            keep.append(False)
            paths.append("")

    missing = int((np.array(keep) == False).sum())
    if missing > 0:
        print(f"[WARNING] Dropping {missing} samples with missing MRI files")

    df = df[np.array(keep)].copy().reset_index(drop=True)
    df["image"] = [p for p in paths if p]

    if args.max_samples and args.max_samples > 0 and len(df) > args.max_samples:
        df = df.head(args.max_samples).copy().reset_index(drop=True)

    return df


def make_participants_df(args: argparse.Namespace, image_index: dict) -> pd.DataFrame:
    participants_tsv = Path(args.bids_root) / "participants.tsv"
    if not participants_tsv.exists():
        raise FileNotFoundError(f"participants.tsv not found: {participants_tsv}")

    df = pd.read_csv(participants_tsv, sep="\t")
    if "participant_id" not in df.columns or "label" not in df.columns:
        raise ValueError("participants.tsv must contain 'participant_id' and 'label' columns")

    df = df[df["label"].notnull()].copy()
    df["sub_id"] = df["participant_id"].astype(str)
    df["true_label"] = df["label"].astype(int)

    paths = []
    keep = []
    for _, row in df.iterrows():
        sub_id = row["sub_id"]
        img_path = Path(args.bids_root) / f"sub-{sub_id}" / "anat" / f"{sub_id}_T1w.nii.gz"
        if sub_id in image_index:
            keep.append(True)
            paths.append(str(image_index[sub_id]))
        elif img_path.exists():
            keep.append(True)
            paths.append(str(img_path))
        else:
            keep.append(False)
            paths.append("")

    missing = int((np.array(keep) == False).sum())
    if missing > 0:
        print(f"[WARNING] Dropping {missing} participants with missing MRI files")

    df = df[np.array(keep)].copy().reset_index(drop=True)
    df["image"] = [p for p in paths if p]
    df = df[["sub_id", "true_label", "image"]]

    if args.max_samples and args.max_samples > 0 and len(df) > args.max_samples:
        df = df.head(args.max_samples).copy().reset_index(drop=True)

    return df


def build_loader(df: pd.DataFrame, target_size: Tuple[int, int, int], batch_size: int, num_workers: int) -> DataLoader:
    data_dicts = [
        {
            "image": row["image"],
            "label": int(row["true_label"]),
            "sub_id": str(row["sub_id"]),
        }
        for _, row in df.iterrows()
    ]

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Resized(keys=["image"], spatial_size=target_size, mode="trilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            EnsureTyped(keys=["image"]),
        ]
    )

    dataset = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def predictive_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1 - 1e-8)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def run_mc_dropout(model: nn.Module, loader: DataLoader, device: torch.device, mc_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_true: List[int] = []
    sub_ids: List[str] = []
    probs_passes: List[np.ndarray] = []

    for s in range(mc_samples):
        enable_dropout_at_inference(model)
        probs_this_pass: List[float] = []

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(device)
                logits = model(images)
                probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                probs_this_pass.extend(probs.tolist())

                if s == 0:
                    y_true.extend(batch["label"].cpu().numpy().astype(int).tolist())
                    sub_ids.extend(list(batch["sub_id"]))

        probs_passes.append(np.array(probs_this_pass, dtype=np.float32))

    probs_matrix = np.stack(probs_passes, axis=0)  # (mc_samples, n_subjects)
    mean_prob = probs_matrix.mean(axis=0)
    std_prob = probs_matrix.std(axis=0)

    pe_mean = predictive_entropy(mean_prob)
    pe_each = predictive_entropy(np.clip(probs_matrix, 1e-8, 1 - 1e-8))
    mean_pe = pe_each.mean(axis=0)
    mutual_info = pe_mean - mean_pe

    return np.array(y_true), np.array(sub_ids), mean_prob, std_prob, mutual_info


def compute_metrics(y_true: np.ndarray, mean_prob: np.ndarray, pred_label: np.ndarray) -> dict:
    auc = float(roc_auc_score(y_true, mean_prob)) if len(np.unique(y_true)) > 1 else float("nan")
    acc = float(accuracy_score(y_true, pred_label))
    f1 = float(f1_score(y_true, pred_label, zero_division=0))
    cm = confusion_matrix(y_true, pred_label, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    return {
        "AUC": auc,
        "Accuracy": acc,
        "F1": f1,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.runs_root) / f"mc_dropout_bnn_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MC DROPOUT BNN INFERENCE (ResNet18)")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Source: {args.source}")
    print(f"Split: {args.split}")
    print(f"MC samples: {args.mc_samples}")
    print(f"Output dir: {out_dir}")

    print("Indexing MRI files under BIDS root...")
    image_index = build_image_index(Path(args.bids_root))
    print(f"Indexed subjects with MRI: {len(image_index)}")

    if args.source == "split":
        split_df = make_split_df(args, image_index=image_index)
    else:
        split_df = make_participants_df(args, image_index=image_index)

    if len(split_df) == 0:
        raise RuntimeError("No valid samples found after matching IDs to MRI files.")
    print(f"Subjects to evaluate: {len(split_df)}")

    loader = build_loader(
        split_df,
        target_size=(args.target_size[0], args.target_size[1], args.target_size[2]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = build_resnet18(dropout_rate=args.dropout_rate, device=device)
    loaded_ckpt, load_logs = load_checkpoint_with_fallback(model, args)
    for msg in load_logs:
        print(msg)

    y_true, sub_ids, mean_prob, std_prob, mutual_info = run_mc_dropout(
        model=model,
        loader=loader,
        device=device,
        mc_samples=args.mc_samples,
    )

    pred_label = (mean_prob > 0.5).astype(int)
    uncertain = (std_prob >= args.uncertainty_threshold).astype(int)

    base_metrics = compute_metrics(y_true, mean_prob, pred_label)

    confident_mask = uncertain == 0
    if confident_mask.sum() > 0:
        conf_metrics = compute_metrics(y_true[confident_mask], mean_prob[confident_mask], pred_label[confident_mask])
    else:
        conf_metrics = {"AUC": float("nan"), "Accuracy": float("nan"), "F1": float("nan"), "Sensitivity": float("nan"), "Specificity": float("nan")}

    results = pd.DataFrame(
        {
            "sub_id": sub_ids,
            "true_label": y_true,
            "mean_prob": mean_prob,
            "pred_label": pred_label,
            "std_prob": std_prob,
            "mutual_info": mutual_info,
            "is_uncertain": uncertain,
        }
    )
    results.to_csv(out_dir / f"{args.split}_mc_dropout_predictions.csv", index=False)

    summary = {
        "timestamp": timestamp,
        "device": str(device),
        "split": args.split,
        "n_subjects": int(len(results)),
        "mc_samples": int(args.mc_samples),
        "dropout_rate": float(args.dropout_rate),
        "uncertainty_threshold": float(args.uncertainty_threshold),
        "checkpoint_used": loaded_ckpt,
        "base_metrics": base_metrics,
        "uncertainty_summary": {
            "n_uncertain": int(uncertain.sum()),
            "uncertain_fraction": float(uncertain.mean()),
            "mean_std_prob": float(std_prob.mean()),
            "median_std_prob": float(np.median(std_prob)),
        },
        "metrics_on_confident_subset": conf_metrics,
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    metrics_rows = [
        {
            "subset": "all",
            "AUC": base_metrics["AUC"],
            "Accuracy": base_metrics["Accuracy"],
            "F1": base_metrics["F1"],
            "Sensitivity": base_metrics["Sensitivity"],
            "Specificity": base_metrics["Specificity"],
            "n_samples": int(len(results)),
        },
        {
            "subset": "confident_only",
            "AUC": conf_metrics.get("AUC", np.nan),
            "Accuracy": conf_metrics.get("Accuracy", np.nan),
            "F1": conf_metrics.get("F1", np.nan),
            "Sensitivity": conf_metrics.get("Sensitivity", np.nan),
            "Specificity": conf_metrics.get("Specificity", np.nan),
            "n_samples": int(confident_mask.sum()),
        },
    ]
    pd.DataFrame(metrics_rows).to_csv(out_dir / "metrics.csv", index=False)

    print("\n--- Metrics (all subjects) ---")
    print(f"AUC:         {base_metrics['AUC']:.4f}")
    print(f"Accuracy:    {base_metrics['Accuracy']:.4f}")
    print(f"Sensitivity: {base_metrics['Sensitivity']:.4f}")
    print(f"Specificity: {base_metrics['Specificity']:.4f}")
    print(f"F1:          {base_metrics['F1']:.4f}")

    print("\n--- Uncertainty ---")
    print(f"Uncertain subjects: {int(uncertain.sum())}/{len(results)} ({uncertain.mean() * 100:.1f}%)")
    print(f"Mean predictive std: {std_prob.mean():.4f}")

    print(f"\nSaved: {out_dir}")
    print("Files:")
    print(f"  - {(out_dir / f'{args.split}_mc_dropout_predictions.csv').name}")
    print("  - metrics.csv")
    print("  - summary.json")


if __name__ == "__main__":
    main()
