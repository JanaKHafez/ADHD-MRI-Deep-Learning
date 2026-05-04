#!/usr/bin/env python3
"""
train_attention_fusion_simple.py
Simplified version - trains attention network on pre-computed predictions
"""

import os
import json
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score, f1_score, ConfusionMatrixDisplay

import torch
import torch.nn as nn
from torch.optim import Adam

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

RESNET_RUN = "runs/ResNet18_best"
BRAINIAC_RUN = "runs/BrainIAC_best"
RUNS_DIR = "runs"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training hyperparameters
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 200
EARLY_STOP = 30
SAVE_DPI = 150

# ══════════════════════════════════════════════════════════════════════════════
# SET SEED
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# GATING NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class GatingNetwork(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        logits = self.net(x)
        return torch.softmax(logits, dim=1)

# ══════════════════════════════════════════════════════════════════════════════
# METRICS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def sensitivity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn = cm[0, 0]
        return tn / (tn + 1e-8) if tn == 1 else 0.0
    tn, fp, fn, tp = cm.ravel()
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        tn = cm[0, 0]
        return tn / (tn + 1e-8) if tn == 1 else 0.0
    tn, fp, fn, tp = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("SIMPLIFIED ATTENTION FUSION - GATING NETWORK TRAINING")
    print("=" * 80)
    print(f"\nDevice: {DEVICE}\n")
    
    # Create output directory
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(RUNS_DIR, f"attention_fusion_{run_timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Load predictions
    # ══════════════════════════════════════════════════════════════════════════
    
    print("[1] Loading predictions...")
    
    # Load all predictions
    resnet_train = pd.read_csv(os.path.join(RESNET_RUN, "train_predictions.csv")).set_index("sub_id")
    resnet_val = pd.read_csv(os.path.join(RESNET_RUN, "val_predictions.csv")).set_index("sub_id")
    resnet_test = pd.read_csv(os.path.join(RESNET_RUN, "test_predictions.csv")).set_index("sub_id")
    
    brainiac_train = pd.read_csv(os.path.join(BRAINIAC_RUN, "train_predictions.csv")).set_index("sub_id")
    brainiac_val = pd.read_csv(os.path.join(BRAINIAC_RUN, "val_predictions.csv")).set_index("sub_id")
    brainiac_test = pd.read_csv(os.path.join(BRAINIAC_RUN, "test_predictions.csv")).set_index("sub_id")
    
    print(f"  Train: {len(resnet_train)} subjects")
    print(f"  Val:   {len(resnet_val)} subjects")
    print(f"  Test:  {len(resnet_test)} subjects\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Prepare features (simplest case: just the two probabilities)
    # ══════════════════════════════════════════════════════════════════════════
    
    print("[2] Preparing features...")
    
    # Use just the two model probabilities as features
    X_train = np.column_stack([
        resnet_train["pred_prob"].values,
        brainiac_train["pred_prob"].values,
    ])
    y_train = resnet_train["true_label"].values
    
    X_val = np.column_stack([
        resnet_val["pred_prob"].values,
        brainiac_val["pred_prob"].values,
    ])
    y_val = resnet_val["true_label"].values
    
    X_test = np.column_stack([
        resnet_test["pred_prob"].values,
        brainiac_test["pred_prob"].values,
    ])
    y_test = resnet_test["true_label"].values
    sub_ids_test = resnet_test.index.values
    
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Train shape: {X_train_scaled.shape}")
    print(f"  Val shape:   {X_val_scaled.shape}")
    print(f"  Test shape:  {X_test_scaled.shape}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Train gating network
    # ══════════════════════════════════════════════════════════════════════════
    
    print("[3] Training gating network...")
    
    model = GatingNetwork(n_input=2).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCELoss()
    
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).to(DEVICE)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    best_auc = -1
    best_state = None
    no_improve = 0
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with proper tensor operations
        weights = model(X_train_t)  # (N, 2)
        p_resnet_t = torch.tensor(X_train[:, 0], dtype=torch.float32).to(DEVICE)
        p_brainiac_t = torch.tensor(X_train[:, 1], dtype=torch.float32).to(DEVICE)
        p_fused = weights[:, 0] * p_resnet_t + weights[:, 1] * p_brainiac_t
        
        # Loss
        loss = criterion(p_fused, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            weights_val = model(X_val_t)
            p_resnet_val = X_val[:, 0]
            p_brainiac_val = X_val[:, 1]
            p_fused_val = weights_val[:, 0].cpu().numpy() * p_resnet_val + weights_val[:, 1].cpu().numpy() * p_brainiac_val
            
            try:
                val_auc = roc_auc_score(y_val, p_fused_val)
            except Exception:
                val_auc = 0.5
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{EPOCHS} | Loss: {loss.item():.4f} | Val AUC: {val_auc:.4f}")
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= EARLY_STOP:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    print(f"\n  Best Val AUC: {best_auc:.4f}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Evaluate on test set
    # ══════════════════════════════════════════════════════════════════════════
    
    print("[4] Evaluating on test set...")
    
    model.eval()
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    
    with torch.no_grad():
        weights_test = model(X_test_t)
        w_resnet = weights_test[:, 0].cpu().numpy()
        w_brainiac = weights_test[:, 1].cpu().numpy()
        
        p_resnet_test = X_test[:, 0]
        p_brainiac_test = X_test[:, 1]
        p_fused_test = w_resnet * p_resnet_test + w_brainiac * p_brainiac_test
    
    y_pred_test = (p_fused_test > 0.5).astype(int)
    
    # Metrics
    test_auc = roc_auc_score(y_test, p_fused_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_sens = sensitivity(y_test, y_pred_test)
    test_spec = specificity(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)
    
    print(f"\n  Test AUC:        {test_auc:.4f}")
    print(f"  Test Accuracy:   {test_acc:.4f}")
    print(f"  Test Sensitivity: {test_sens:.4f}")
    print(f"  Test Specificity: {test_spec:.4f}")
    print(f"  Test F1 Score:   {test_f1:.4f}\n")
    
    # ══════════════════════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════════════════════
    
    print("[5] Saving results...")
    
    # Predictions CSV
    results_df = pd.DataFrame({
        "sub_id": sub_ids_test,
        "true_label": y_test,
        "pred_prob": p_fused_test,
        "pred_label": y_pred_test,
        "w_resnet": w_resnet,
        "w_brainiac": w_brainiac,
    })
    results_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    # Metrics CSV
    metrics_df = pd.DataFrame([{
        "variant": "gating_network",
        "AUC": test_auc,
        "Accuracy": test_acc,
        "Sensitivity": test_sens,
        "Specificity": test_spec,
        "F1": test_f1,
    }])
    metrics_df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Control", "ADHD"])
    disp.plot(ax=ax)
    ax.set_title("Gating Network - Test Set Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=SAVE_DPI)
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, p_fused_test)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    ax.plot(fpr, tpr, label=f"Gating Network (AUC={test_auc:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=SAVE_DPI)
    plt.close()
    
    # Weights by label
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    control_w_resnet = w_resnet[y_test == 0].mean()
    control_w_brainiac = w_brainiac[y_test == 0].mean()
    adhd_w_resnet = w_resnet[y_test == 1].mean()
    adhd_w_brainiac = w_brainiac[y_test == 1].mean()
    
    x = np.arange(2)
    width = 0.35
    ax.bar(x - width/2, [control_w_resnet, adhd_w_resnet], width, label="ResNet18", color="steelblue")
    ax.bar(x + width/2, [control_w_brainiac, adhd_w_brainiac], width, label="BrainIAC", color="coral")
    ax.set_ylabel("Mean Weight")
    ax.set_title("Gating Weights by True Label")
    ax.set_xticks(x)
    ax.set_xticklabels(["Control", "ADHD"])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "weights_by_label.png"), dpi=SAVE_DPI)
    plt.close()
    
    # Hyperparameters
    hyper = {
        "timestamp": run_timestamp,
        "device": str(DEVICE),
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "epochs_trained": epoch + 1,
        "early_stopping_patience": EARLY_STOP,
        "best_val_auc": float(best_auc),
    }
    with open(os.path.join(output_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyper, f, indent=4)
    
    print(f"\n  Results saved to {output_dir}\n")
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
