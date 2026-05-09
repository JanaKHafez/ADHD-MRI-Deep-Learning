# ADHD Classification from Structural MRI with Uncertainty Quantification

Deep learning pipeline for ADHD classification from sMRI scans, featuring multiple model architectures and uncertainty estimation methods.

## Project Overview

This repository implements end-to-end ADHD diagnosis classification using 3D brain MRI images. It includes:
- Multiple CNN architectures (ResNet, BrainIAC, SFCN, DenseNet)
- Ensemble methods with attention-based fusion
- Three uncertainty quantification approaches (EDL, MC Dropout, Gating Network)
- Complete preprocessing, training, and evaluation pipeline

## Project Structure

### Data Preparation
- **`prepare_adhd_bids.py`** - Converts raw ADHD200 data to BIDS format with standardized participants metadata
- **`src/sMRI_adhd_pipeline.py`** - Main preprocessing pipeline: registration, skull stripping, normalization
- **`src/segment.py`** - Brain segmentation using SynthSeg

### Notebooks
- **`src/Dataset.ipynb`** - Data loading and exploration
- **`src/EDA.ipynb`** - Exploratory data analysis and statistics
- **`src/site_diag.ipynb`** - Site-level diagnostics and class balance analysis
- **`src/region_saliency.ipynb`** - Brain region importance visualization
- **`src/Linear_SVM.ipynb`** - SVM baseline model
- **`src/inference.ipynb`** - Model inference and prediction

### Training & Inference
- **`src/sMRI_adhd_pipeline.py`** - End-to-end training script for standard models
- **`src/ensemble.py`** - Ensemble voting and aggregation methods

### Uncertainty Quantification
- **`src/evidential_deep_learning_simple.py`** - Evidential Deep Learning (EDL) using Dirichlet distributions
- **`src/mc_dropout_bnn_resnet18.py`** - MC Dropout Bayesian approximation for ResNet18
- **`src/train_attention_fusion_simple.py`** - Attention-based gating network for ensemble fusion

## Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

For segmentation (SynthSeg):
```bash
pip install -r segment_requirements.txt
```

### 2. Data Preparation
```bash
python prepare_adhd_bids.py
python src/sMRI_adhd_pipeline.py --bids-root ADHD_BIDS
```

### 3. Training
Run notebooks in order or use scripts:
```bash
python src/sMRI_adhd_pipeline.py --mode train
python src/train_attention_fusion_simple.py
python src/evidential_deep_learning_simple.py --mode train
```

### 4. Uncertainty Estimation
```bash
# MC Dropout
python src/mc_dropout_bnn_resnet18.py --split test --mc-samples 30

# Evidential Deep Learning
python src/evidential_deep_learning_simple.py --mode inference --split test

# Attention Fusion
python src/train_attention_fusion_simple.py
```

## Model Architectures

| Model | Input | Output | Use Case |
|-------|-------|--------|----------|
| ResNet18/50 | 3D MRI (121×128×121) | Class probability | General classification |
| BrainIAC | 3D MRI | Attention map + Class | Interpretability |
| SFCN | 3D MRI | Regression output | Age/sex prediction |
| DenseNet121 | 3D MRI | Class probability | Alternative backbone |

## Uncertainty Methods

1. **Evidential Deep Learning (EDL)**
   - Learns Dirichlet distribution parameters
   - Estimates aleatoric and epistemic uncertainty
   - File: `src/evidential_deep_learning_simple.py`

2. **MC Dropout**
   - Monte Carlo sampling with dropout at inference
   - Bayesian approximation of neural networks
   - File: `src/mc_dropout_bnn_resnet18.py`

3. **Attention Fusion**
   - Learned gating weights for ensemble predictions
   - Combines ResNet18 + BrainIAC outputs
   - File: `src/train_attention_fusion_simple.py`

## Output Structure

```
runs/
├── ResNet18_best/          # Best ResNet18 model
├── BrainIAC_best/          # Best BrainIAC model
├── ensemble/               # Ensemble predictions
├── edl_brainiac_*/         # EDL results
├── mc_dropout_bnn_*/       # MC Dropout results
└── attention_fusion_*/     # Gating network results

output/
├── eda_results_ADHD/       # EDA statistics
├── eda_results_gender/     # Gender analysis
├── site_diagnostics/       # Site-level metrics
└── brain_3d.html           # 3D brain visualization
```

## Key Files Reference

| File | Purpose |
|------|---------|
| `requirements.txt` | Python dependencies |
| `segment_requirements.txt` | SynthSeg dependencies |
| `pretrained_models/` | Pretrained weights (ResNet, SFCN) |
| `src/BrainIAC-main/` | BrainIAC model implementation |
| `src/SynthSeg/` | SynthSeg brain segmentation |

## Results

Results are saved in `runs/` with standard outputs:
- `train_predictions.csv` - Predictions on training set
- `val_predictions.csv` - Validation predictions
- `test_predictions.csv` - Test set results
- `metrics.csv` - Performance metrics
- `summary.json` - Training summary

## Citation

If you use this work, please cite the ADHD200 dataset and relevant methods:
- ADHD200: http://www.adhd200.org/
- BrainIAC: [original paper]
- EDL: Amini et al. (2020) "Uncertainty Quantification 360° in Deep Learning"
