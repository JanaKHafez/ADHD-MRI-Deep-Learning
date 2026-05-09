
AVAILABLE MODELS
================================================================================

PRETRAINED MODELS (pretrained_models/ directory)
--------
These are pre-trained foundation models used for transfer learning:

1. ResNet18 (resnet_18.pth)
   - Architecture: 3D ResNet-18
   - Size: ~45 MB
   - Usage: Feature extractor, classifier backbone
   - Source: MedicalNet pretrained weights
   - Input: 3D MRI (121x128x121)

2. ResNet50 (resnet_50.pth)
   - Architecture: 3D ResNet-50
   - Size: ~98 MB
   - Usage: Deeper feature extraction
   - Source: MedicalNet pretrained weights
   - Input: 3D MRI (121x128x121)

3. BrainIAC (BrainIAC.ckpt)
   - Architecture: Attention-based CNN
   - Size: ~150 MB
   - Usage: Interpretable classification with saliency maps
   - Input: 3D MRI (121x128x121)

4. BrainIAC MCI (BrainIAC_mci.ckpt)
   - Variant for MCI (Mild Cognitive Impairment) detection
   - Size: ~150 MB
   - Input: 3D MRI (121x128x121)

5. SFCN Models (SFCN_age.p, SFCN_sex.p)
   - Architecture: Spatial FCN for demographic prediction
   - Age prediction: SFCN_age.p
   - Sex prediction: SFCN_sex.p
   - Usage: Auxiliary demographic inference


TRAINED BEST MODELS (runs/ directory)
================================================================================

Best-performing trained models from our experiments:

BRAINIAC_BEST
   Location: runs/BrainIAC_best/
   Trained on: ADHD200 dataset
   Output files:
   - test_predictions.csv - Predictions on test set
   - val_predictions.csv - Validation predictions
   - train_predictions.csv - Training predictions
   - metrics.csv - Performance metrics
   - hyperparameters.json - Training configuration
   - confusion_matrix.png - Visualization
   
   Performance (see metrics.csv):
   - Test AUC: [From metrics.csv]
   - Test Accuracy: [From metrics.csv]
   - Sensitivity: [From metrics.csv]
   - Specificity: [From metrics.csv]
   
   Note: Model weights from this run are stored in predictions.
         Can be reproduced by retraining with same hyperparameters.

RESNET18_BEST
   Location: runs/ResNet18_best/
   Trained on: ADHD200 dataset
   Output files:
   - test_predictions.csv - Predictions on test set
   - val_predictions.csv - Validation predictions
   - train_predictions.csv - Training predictions
   - metrics.csv - Performance metrics
   - hyperparameters.json - Training configuration
   - confusion_matrix.png - Visualization
   
   Performance (see metrics.csv):
   - Test AUC: [From metrics.csv]
   - Test Accuracy: [From metrics.csv]
   - Sensitivity: [From metrics.csv]
   - Specificity: [From metrics.csv]
   
   Note: Model weights from this run are stored in predictions.
         Can be reproduced by retraining with same hyperparameters.


UNCERTAINTY QUANTIFICATION MODELS
================================================================================

Models trained for uncertainty estimation:

EVIDENTIAL DEEP LEARNING (EDL)
   Script: src/evidential_deep_learning_simple.py
   Input: Ensemble predictions from ResNet18 + BrainIAC
   Output: Uncertainty estimates + calibrated confidence
   Output: runs/edl_brainiac_*/
   
   Outputs:
   - test_predictions.csv - Predictions with uncertainty
   - summary.json - Training summary

MC DROPOUT BAYESIAN
   Script: src/mc_dropout_bnn_resnet18.py
   Input: ResNet18 with dropout at inference
   Output: Uncertainty via stochastic forward passes
   Output: runs/mc_dropout_bnn_*/
   
   Outputs:
   - test_mc_dropout_predictions.csv - MC predictions
   - metrics.csv - Calibrated metrics
   - summary.json - Uncertainty statistics

ATTENTION FUSION GATING
   Script: src/train_attention_fusion_simple.py
   Input: ResNet18 + BrainIAC ensemble
   Output: Learned weighted fusion with uncertainty
   Output: runs/attention_fusion_*/
   
   Outputs:
   - test_predictions.csv - Fused predictions
   - metrics.csv - Ensemble metrics


REPRODUCIBILITY
================================================================================

To reproduce trained models:

1. Verify Data:
   python prepare_adhd_bids.py
   python src/sMRI_adhd_pipeline.py --mode preprocess

2. Train ResNet18:
   python src/sMRI_adhd_pipeline.py --mode train --model resnet18
   
   Hyperparameters (from runs/ResNet18_best/hyperparameters.json):
   - Learning rate: [from JSON]
   - Batch size: [from JSON]
   - Epochs: [from JSON]
   - Early stopping patience: [from JSON]

3. Train BrainIAC:
   python src/sMRI_adhd_pipeline.py --mode train --model brainiac
   
   Hyperparameters (from runs/BrainIAC_best/hyperparameters.json):
   - Learning rate: [from JSON]
   - Batch size: [from JSON]
   - Epochs: [from JSON]
   - Early stopping patience: [from JSON]

4. Train Uncertainty Methods:
   python src/evidential_deep_learning_simple.py --mode train
   python src/mc_dropout_bnn_resnet18.py
   python src/train_attention_fusion_simple.py


MODEL PERFORMANCE SUMMARY
================================================================================

See performance metrics in:
  runs/ResNet18_best/metrics.csv
  runs/BrainIAC_best/metrics.csv
  runs/edl_brainiac_*/summary.json
  runs/mc_dropout_bnn_*/metrics.csv
  runs/attention_fusion_*/metrics.csv

Key metrics tracked:
  - AUC (Area Under ROC Curve)
  - Accuracy
  - Sensitivity (True Positive Rate)
  - Specificity (True Negative Rate)
  - F1 Score
  - Confusion Matrix


DOWNLOADING WEIGHTS
================================================================================

All model weights are included in the repository:

Pretrained models:
  Location: pretrained_models/
  Size: ~450 MB total
  Status: Included in repository

If downloading separately:
  https://github.com/JanaKHafez/ADHD-MRI-Deep-Learning/releases/download/v1.0/pretrained_models.zip


LOADING MODELS
================================================================================

In Python:

import torch
from src.sMRI_adhd_pipeline import build_model

# Load pretrained model
model = build_model('resnet18')
model.load_state_dict(torch.load('pretrained_models/resnet_18.pth'))
model.eval()

# Load trained best model (retrain or use predictions from best run)
# Predictions are in: runs/ResNet18_best/test_predictions.csv


INFERENCE
================================================================================

Using saved predictions from best models:

import pandas as pd

# Load BrainIAC best predictions
predictions = pd.read_csv('runs/BrainIAC_best/test_predictions.csv')
print(predictions.head())

# Load ResNet18 best predictions
predictions = pd.read_csv('runs/ResNet18_best/test_predictions.csv')
print(predictions.head())

# Predictions include:
#   - sub_id: Subject ID
#   - true_label: Ground truth label
#   - pred_prob: Predicted probability for ADHD class
#   - pred_label: Predicted class (0=Control, 1=ADHD)


NOTES
================================================================================

1. All hyperparameters used for training are saved in hyperparameters.json
   in each run directory.

3. To obtain the trained model weights, retrain using the saved hyperparameters:
   python src/sMRI_adhd_pipeline.py --mode train \
     --config runs/ResNet18_best/hyperparameters.json

4. Uncertainty estimates are produced by dedicated scripts:
   - src/evidential_deep_learning_simple.py
   - src/mc_dropout_bnn_resnet18.py
   - src/train_attention_fusion_simple.py

5. All predictions are reproducible given the same data and hyperparameters.
