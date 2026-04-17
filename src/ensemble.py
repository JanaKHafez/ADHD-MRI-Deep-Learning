# ensemble_advanced.py
# ══════════════════════════════════════════════════════════════════════════════
# Advanced Ensemble: ResNet18 + BrainIAC
# Strategies: simple average | rank average | OWA (val-AUC weights) |
#             logistic stacking | Bayesian weight optimisation
# Calibration: Platt scaling or Isotonic Regression applied before ensembling
# ══════════════════════════════════════════════════════════════════════════════

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from itertools import product
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
RESNET18_RUN_DIR = r"runs/ResNet18_best"
BRAINIAC_RUN_DIR = r"runs/BrainIAC_best"
OUT_DIR          = "runs/ensemble"

# Calibration method applied to each model's raw probabilities before ensembling.
# "platt"    : Platt scaling — fits a logistic regression on val probs (best when
#              the model is systematically overconfident, i.e. probs cluster at 0/1).
# "isotonic" : Isotonic regression — non-parametric, more flexible, needs more val
#              data (≥100 samples). Tends to overfit on tiny validation sets.
# None       : No calibration — use raw softmax outputs as-is.
CALIBRATION      = None   # "platt" | "isotonic" | None

FIND_OPTIMAL_THRESHOLD = True   # Youden's J on ensemble val probs → test threshold
SAVE_DPI                = 150
# ══════════════════════════════════════════════════════════════════════════════


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_split(run_dir, split):
    path = os.path.join(run_dir, f"{split}_predictions.csv")
    df   = pd.read_csv(path).set_index("sub_id")
    return df[["true_label", "pred_prob"]]


def align(df_a, df_b):
    """Inner-join two prediction frames on sub_id; assert label consistency."""
    m = df_a.join(df_b, how="inner", lsuffix="_a", rsuffix="_b")
    assert (m["true_label_a"] == m["true_label_b"]).all(), \
        "Label mismatch between models — check that splits are identical."
    ids    = m.index.values
    labels = m["true_label_a"].values.astype(int)
    p_a    = m["pred_prob_a"].values
    p_b    = m["pred_prob_b"].values
    return ids, labels, p_a, p_b


# ── Calibration ───────────────────────────────────────────────────────────────

class PlattCalibrator:
    """
    Platt scaling: fits a 1-D logistic regression mapping raw model
    probabilities → calibrated probabilities, using the validation set.
    Very sample-efficient; works well even with <50 val points.
    """
    def fit(self, val_probs, val_labels):
        X = val_probs.reshape(-1, 1)
        self.lr = LogisticRegression(C=1e10)   # effectively no regularisation
        self.lr.fit(X, val_labels)
        return self

    def predict(self, probs):
        return self.lr.predict_proba(probs.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """
    Isotonic regression calibration: fits a monotone step function from
    raw probs → calibrated probs. More flexible than Platt but needs more
    validation data and can overfit on tiny sets.
    """
    def fit(self, val_probs, val_labels):
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.ir.fit(val_probs, val_labels.astype(float))
        return self

    def predict(self, probs):
        return self.ir.predict(probs)


def calibrate(val_probs, val_labels, test_probs, method):
    if method == "platt":
        cal = PlattCalibrator().fit(val_probs, val_labels)
    elif method == "isotonic":
        cal = IsotonicCalibrator().fit(val_probs, val_labels)
    else:
        return val_probs, test_probs

    return cal.predict(val_probs), cal.predict(test_probs)


# ── Ensemble strategies ───────────────────────────────────────────────────────

def strategy_mean(p_a, p_b, **_):
    """Simple unweighted average of calibrated probabilities."""
    return 0.5 * p_a + 0.5 * p_b


def strategy_rank(p_a, p_b, **_):
    """
    Rank-based ensemble: convert each model's probabilities to percentile
    ranks, then average the ranks.  More robust than averaging raw probs
    when the two models' probability scales differ significantly even after
    calibration — ranks are always in [0, 1] by construction.
    """
    n     = len(p_a)
    rank_a = rankdata(p_a) / n
    rank_b = rankdata(p_b) / n
    return 0.5 * rank_a + 0.5 * rank_b


def strategy_owa(p_a, p_b, val_true, val_p_a, val_p_b, **_):
    """
    Optimally Weighted Average (OWA): assign weights proportional to each
    model's individual validation AUC, then re-normalise so they sum to 1.
    A model that contributes twice the AUC lift gets twice the weight.
    """
    auc_a = roc_auc_score(val_true, val_p_a)
    auc_b = roc_auc_score(val_true, val_p_b)
    w_a   = auc_a / (auc_a + auc_b)
    w_b   = auc_b / (auc_a + auc_b)
    print(f"  [OWA] ResNet18 w={w_a:.3f}  BrainIAC w={w_b:.3f}  "
          f"(val AUCs: {auc_a:.4f} / {auc_b:.4f})")
    return w_a * p_a + w_b * p_b


def strategy_stacking(p_a, p_b, val_true, val_p_a, val_p_b, **_):
    """
    Logistic Stacking (level-1 meta-learner): trains a logistic regression
    on the *validation* probabilities of both models and uses it to predict
    the test labels.  Can discover non-linear combinations — e.g., trust
    BrainIAC more when ResNet18 is uncertain (prob near 0.5).

    ⚠ Requires enough validation data for the meta-learner not to overfit.
    With < 40 val samples, consider cross-validated stacking instead.
    """
    X_val  = np.column_stack([val_p_a, val_p_b])
    X_test = np.column_stack([p_a, p_b])
    meta   = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(X_val, val_true)
    coef   = meta.coef_[0]
    print(f"  [Stacking] Meta-learner coefs: ResNet18={coef[0]:.3f}  BrainIAC={coef[1]:.3f}")
    return meta.predict_proba(X_test)[:, 1]


def strategy_bayesian_opt(p_a, p_b, val_true, val_p_a, val_p_b, **_):
    """
    Nelder-Mead weight optimisation: searches for the scalar α ∈ [0,1] that
    maximises validation AUC for the blend α·p_a + (1-α)·p_b.

    Uses scipy.optimize.minimize with the Nelder-Mead method, which is
    derivative-free and handles non-smooth objectives like AUC well.
    The search is constrained to [0,1] by reflecting out-of-bounds α back.
    """
    def neg_auc(alpha):
        # Extract the scalar value from the 1D array using alpha[0]
        alpha = float(np.clip(alpha[0], 0.0, 1.0))
        blend = alpha * val_p_a + (1 - alpha) * val_p_b
        return -roc_auc_score(val_true, blend)

    result = minimize(neg_auc, x0=[0.5], method="Nelder-Mead",
                      options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 500})
    alpha  = float(np.clip(result.x[0], 0.0, 1.0))
    print(f"  [BayesOpt] Optimal α={alpha:.4f} → "
          f"ResNet18={alpha:.3f}  BrainIAC={1-alpha:.3f}  "
          f"val AUC={-result.fun:.4f}")
    return alpha * p_a + (1 - alpha) * p_b


STRATEGIES = {
    "mean":       strategy_mean,
    "rank":       strategy_rank,
    "owa":        strategy_owa,
    "stacking":   strategy_stacking,
    "bayesian":   strategy_bayesian_opt,
}


# ── Evaluation helpers ────────────────────────────────────────────────────────

def find_threshold(val_true, val_probs):
    fpr, tpr, thresholds = roc_curve(val_true, val_probs)
    j_idx = np.argmax(tpr - fpr)
    return float(thresholds[j_idx])


def evaluate(name, true, probs, threshold):
    preds     = (probs > threshold).astype(int)
    auc       = roc_auc_score(true, probs)
    acc       = np.mean(preds == true)
    cm        = confusion_matrix(true, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec      = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    print(f"  {name:<22s}  AUC={auc:.4f}  Acc={acc:.4f}  "
          f"Sens={sens:.4f}  Spec={spec:.4f}  (τ={threshold:.3f})")
    return dict(strategy=name, AUC=auc, Accuracy=acc,
                Sensitivity=sens, Specificity=spec,
                Threshold=threshold, TP=int(tp), TN=int(tn),
                FP=int(fp), FN=int(fn))


def plot_calibration(val_true, raw_probs_dict, cal_probs_dict, save_path):
    """Reliability diagram before/after calibration for each model."""
    n_models = len(raw_probs_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5), facecolor="white")
    if n_models == 1:
        axes = [axes]
    for ax, (name, raw) in zip(axes, raw_probs_dict.items()):
        cal = cal_probs_dict[name]
        for probs, label, ls in [(raw, "before calibration", "--"),
                                  (cal, "after calibration",  "-")]:
            frac_pos, mean_pred = calibration_curve(val_true, probs, n_bins=10)
            ax.plot(mean_pred, frac_pos, marker="o", ls=ls, label=label)
        ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
        ax.set_title(f"{name} calibration"); ax.set_xlabel("Mean predicted prob")
        ax.set_ylabel("Fraction positives"); ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=SAVE_DPI)
    plt.close(fig)
    print(f"  Calibration plot saved → {save_path}")


def plot_roc_comparison(test_true, probs_dict, save_path):
    plt.figure(figsize=(7, 6), facecolor="white")
    for name, probs in probs_dict.items():
        fpr, tpr, _ = roc_curve(test_true, probs)
        auc = roc_auc_score(test_true, probs)
        lw  = 2.5 if "ensemble" in name else 1.5
        ls  = "-"  if "ensemble" in name else "--"
        plt.plot(fpr, tpr, lw=lw, ls=ls, label=f"{name}  (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — all strategies vs individual models")
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=SAVE_DPI)
    plt.close()
    print(f"  ROC plot saved → {save_path}")


def plot_prob_scatter(test_true, p_a, p_b, save_path):
    """
    Scatter plot of ResNet18 probs vs BrainIAC probs, coloured by true label.
    Points near the diagonal → models agree; off-diagonal → they disagree.
    Disagreements in the correct direction (one model compensates) are where
    ensembling adds most value.
    """
    colours = np.where(test_true == 1, "#D85A30", "#185FA5")
    plt.figure(figsize=(6, 6), facecolor="white")
    plt.scatter(p_a, p_b, c=colours, alpha=0.7, edgecolors="none", s=35)
    plt.plot([0, 1], [0, 1], "k:", lw=1)
    plt.xlabel("ResNet18 probability"); plt.ylabel("BrainIAC probability")
    plt.title("Model agreement (coral=ADHD, blue=Control)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=SAVE_DPI)
    plt.close()
    print(f"  Scatter plot saved → {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load saved prediction CSVs
    print("Loading saved predictions…")
    rn_val   = load_split(RESNET18_RUN_DIR, "val")
    rn_test  = load_split(RESNET18_RUN_DIR, "test")
    bi_val   = load_split(BRAINIAC_RUN_DIR, "val")
    bi_test  = load_split(BRAINIAC_RUN_DIR, "test")

    val_ids,  val_true,  raw_val_rn,  raw_val_bi  = align(rn_val,  bi_val)
    test_ids, test_true, raw_test_rn, raw_test_bi = align(rn_test, bi_test)

    print(f"  Val  set: {len(val_true)} subjects  "
          f"({val_true.sum()} ADHD / {(val_true==0).sum()} Control)")
    print(f"  Test set: {len(test_true)} subjects  "
          f"({test_true.sum()} ADHD / {(test_true==0).sum()} Control)")

    # 2. Calibration
    print(f"\nCalibrating model outputs ({CALIBRATION or 'none'})…")
    val_rn,  test_rn  = calibrate(raw_val_rn,  val_true, raw_test_rn,  CALIBRATION)
    val_bi,  test_bi  = calibrate(raw_val_bi,  val_true, raw_test_bi,  CALIBRATION)

    plot_calibration(
        val_true,
        {"ResNet18": raw_val_rn, "BrainIAC": raw_val_bi},
        {"ResNet18": val_rn,     "BrainIAC": val_bi},
        os.path.join(OUT_DIR, "calibration_curves.png"),
    )

    # 3. Run every ensemble strategy
    print("\nComputing ensemble strategies…")
    shared = dict(val_true=val_true, val_p_a=val_rn, val_p_b=val_bi)
    val_ensembles  = {}
    test_ensembles = {}

    for name, fn in STRATEGIES.items():
        val_ensembles[name]  = fn(val_rn,  val_bi,  **shared)
        test_ensembles[name] = fn(test_rn, test_bi, **shared)

    # 4. Determine threshold for every strategy from the validation ensemble
    print("\nFinding optimal thresholds on validation set (Youden's J)…")
    thresholds = {}
    for name, val_probs in val_ensembles.items():
        thresholds[name] = find_threshold(val_true, val_probs) \
                           if FIND_OPTIMAL_THRESHOLD else 0.5

    # 5. Evaluate on test set
    print("\nTest set evaluation:")
    print(f"  {'strategy':<22s}  AUC       Acc       Sens      Spec      threshold")
    print(f"  {'-'*22}  {'-'*7}   {'-'*7}   {'-'*7}   {'-'*7}   {'-'*9}")

    # Individual models first (baseline)
    threshold_indiv = find_threshold(val_true, val_rn) if FIND_OPTIMAL_THRESHOLD else 0.5
    metrics_rn = evaluate("ResNet18 (baseline)",  test_true, test_rn,  threshold_indiv)
    threshold_indiv_bi = find_threshold(val_true, val_bi) if FIND_OPTIMAL_THRESHOLD else 0.5
    metrics_bi = evaluate("BrainIAC (baseline)",  test_true, test_bi,  threshold_indiv_bi)

    all_metrics = [metrics_rn, metrics_bi]
    for name in STRATEGIES:
        m = evaluate(f"ensemble:{name}", test_true, test_ensembles[name], thresholds[name])
        all_metrics.append(m)

    # 6. Save outputs
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(OUT_DIR, "ensemble_metrics.csv"), index=False)

    # Predictions CSV with every strategy's output
    pred_df = pd.DataFrame({"sub_id": test_ids, "true_label": test_true,
                             "prob_resnet18": test_rn, "prob_brainiac": test_bi})
    for name, probs in test_ensembles.items():
        pred_df[f"prob_{name}"]  = probs
        pred_df[f"pred_{name}"] = (probs > thresholds[name]).astype(int)
    pred_df.to_csv(os.path.join(OUT_DIR, "ensemble_test_predictions.csv"), index=False)

    # ROC overlay
    roc_dict = {"ResNet18": test_rn, "BrainIAC": test_bi}
    roc_dict.update({f"ensemble:{k}": v for k, v in test_ensembles.items()})
    plot_roc_comparison(test_true, roc_dict,
                        os.path.join(OUT_DIR, "ensemble_roc.png"))

    # Model-agreement scatter
    plot_prob_scatter(test_true, test_rn, test_bi,
                      os.path.join(OUT_DIR, "model_agreement_scatter.png"))

    # Summary table
    print(f"\nMetrics saved to: {OUT_DIR}/")
    print("\nSummary (sorted by Test AUC):")
    print(metrics_df[["strategy", "AUC", "Sensitivity", "Specificity"]]
          .sort_values("AUC", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()