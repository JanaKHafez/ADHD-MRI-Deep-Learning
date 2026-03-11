# ══════════════════════════════════════════════════════════════════════════════
# [0] MASTER CONFIGURATION — Fully Annotated
# ══════════════════════════════════════════════════════════════════════════════

# DO NOT REMOVE THIS COMMENTS! This section is crucial for understanding and modifying the code.

TEST = False  # Set to True for a quick test run with a tiny subset of data and epochs.

# ── Paths ─────────────────────────────────────────────────────────────────────
# BIDS_ROOT: The master directory containing your neuroimaging data formatted
# according to the Brain Imaging Data Structure (BIDS) standard.
BIDS_ROOT          = r"ADHD_BIDS"

# RUNS_DIR: The folder where all outputs (best weights, logs, CSVs, and generated
# saliency maps) will be saved. Each run creates a new timestamped subfolder here.
RUNS_DIR           = "runs"

# Pretrained Weight Paths: Paths to locally downloaded checkpoint files.
# Using pre-trained weights (Transfer Learning) significantly reduces training
# time and improves accuracy on smaller medical datasets.
WEIGHT_PATH   = r"pretrained_models\BrainIAC.ckpt"

# ── Model ─────────────────────────────────────────────────────────────────────
# CHOSEN_MODEL: Toggles the neural network architecture.
# - "ResNet18": Standard 3D residual network (good baseline).
# - "ResNet50": Deeper residual network (more capacity, may overfit).
# - "DenseNet121": Densely connected network (good for feature reuse).
# - "SFCN": Simple Fully Convolutional Network (specialised for brain age/classification).
# - "AnatCL": Contrastive learning model pre-trained on diverse brain MRIs.
# - "BrainIAC": ViT-B foundation model pre-trained on 32,000 brain MRIs.
CHOSEN_MODEL       = "AnatCL"  

# NUM_CLASSES: Binary classification (0 = Control/TD, 1 = ADHD).
NUM_CLASSES        = 2

# ── Data & Splitting ──────────────────────────────────────────────────────────
# LABEL_COLUMN: The exact header name in your 'participants.tsv' containing the targets.
LABEL_COLUMN       = "label"      

# TARGET_SIZE: The standardized dimensions all 3D MRIs will be resized to before
# entering the network. (121, 128, 121) is a common standard to balance detail
# and GPU VRAM constraints.
TARGET_SIZE        = (96, 96, 96) if CHOSEN_MODEL == "BrainIAC" else (121, 128, 121)

# VAL_SPLIT / TEST_SPLIT: The proportion of data allocated to Validation and Testing.
# Validation is used to tune hyperparameters and trigger early stopping.
# Test is strictly held out until the very end to evaluate true generalisation.
VAL_SPLIT          = 0.2          
TEST_SPLIT         = 0.1          
RANDOM_STATE       = 42           # Seed for reproducibility.

# USE_SITE_HOLDOUT: If True, it overrides the random TEST_SPLIT. Instead of randomly
# shuffling all patients, it takes an entire hospital/scanner site and holds it out.
# This tests if the model learns generic ADHD features rather than scanner-specific artifacts.
USE_SITE_HOLDOUT   = True        
HOLDOUT_SITE       = "NeuroIMAGE"  # The site name to hold out (must match the 'site' column in participants.tsv).

# ── Advanced Training Options ─────────────────────────────────────────────────
# USE_CACHE_DATASET: If True, MONAI will load all MRIs into RAM on the first epoch.
# Massively speeds up training, but requires a lot of System RAM (32GB+ recommended).
USE_CACHE_DATASET       = True      

# FIND_OPTIMAL_THRESHOLD: If True, uses Youden's J statistic on the validation ROC
# curve to find the best cut-off probability (instead of defaulting to > 0.5).
FIND_OPTIMAL_THRESHOLD  = False      

# EARLY_STOPPING_PATIENCE: Halts training if the Validation AUC doesn't improve
# for this many epochs, preventing the model from memorising the training set (overfitting).
EARLY_STOPPING_PATIENCE = 25        

# SCHEDULER_TYPE: Adjusts the learning rate during training.
# - "Plateau": Drops the learning rate when validation metric stops improving.
# - "Cosine": Gradually smoothly drops the learning rate following a cosine curve.
SCHEDULER_TYPE          = "Cosine"  
SCHEDULER_PATIENCE      = 10         # How many epochs to wait before dropping LR (Plateau).
SCHEDULER_FACTOR        = 0.5        # Multiplier for the LR drop (e.g., halves the LR).
# SCHEDULER_T_MAX: The period of the cosine wave for the CosineAnnealingLR.
# A smaller T_MAX means the learning rate will drop more rapidly and restart more often, 
# which can help escape local minima but may require more epochs to converge.
SCHEDULER_T_MAX  = 20        # Number of epochs for a full cosine cycle (Cosine).

# USE_CLASS_WEIGHTS: If True, penalises misclassification of the minority class
# more heavily by weighting the CrossEntropyLoss inversely to class frequency.
USE_CLASS_WEIGHTS = False

# ── Augmentation (Conservative FSL style) ─────────────────────────────────────
# Data augmentation creates artificial variations of the training images to make
# the model robust. These specific values are kept "conservative" (very subtle)
# because brains pre-processed by FSL are already supposed to be aligned and standardized.
# AUG_AFFINE_PROB: Chance of applying rotation/translation.
AUG_AFFINE_PROB       = 0.3
# AUG_AFFINE_ROTATE: Max rotation in radians (0.03 rad is ~1.7 degrees). Simulates imperfect registration.
AUG_AFFINE_ROTATE     = (0.03, 0.03, 0.03)
# AUG_AFFINE_TRANSLATE: Max pixel shift. Simulates slight bounding box differences.
AUG_AFFINE_TRANSLATE  = (2, 2, 2)

# AUG_NOISE_PROB: Simulates electronic thermal noise from the MRI scanner coil.
AUG_NOISE_PROB        = 0.2
AUG_NOISE_STD         = 0.03      # Standard deviation of the Gaussian noise.

# AUG_SMOOTH_PROB: Simulates slight patient micro-movements (motion blur) during the scan.
AUG_SMOOTH_PROB       = 0.15
AUG_SMOOTH_SIGMA      = (0.25, 0.5) # Smoothing kernel size.

# ── Training ──────────────────────────────────────────────────────────────────
# EPOCHS: Maximum number of times the model will see the entire dataset.
EPOCHS           = 200      
# BATCH_SIZE: Number of 3D MRIs processed simultaneously. 3D images are massive,
# so this is usually kept very small (2 to 8) to avoid "CUDA Out of Memory" errors.
BATCH_SIZE       = 4
# NUM_WORKERS: Number of CPU threads dedicated to loading and augmenting data.
NUM_WORKERS      = 2
# LEARNING_RATE: How large of a step the optimizer takes when updating weights.
# Defult = 1e-5
LEARNING_RATE    = 5e-4
# WEIGHT_DECAY: L2 Regularization term (AdamW). Penalises excessively large weights,
# forcing the network to rely on multiple subtle features rather than one loud noise artifact.
WEIGHT_DECAY     = 1e-2

# ── Fine-tuning Options ───────────────────────────────────────────────────────
# When using transfer learning, we often "freeze" the early layers (which detect
# generic edges/shapes) and only "unfreeze" (train) the deeper layers and the final
# classification head to adapt to our specific ADHD task.
SFCN_FROZEN_BLOCKS      = list(range(4))             # Freezes blocks 0, 1, 2, and 3.
ANATCL_TRAINABLE_LAYERS = ["layer4", "fc"]           # Only trains the last convolutional block and fully connected layer.
RESNET_TRAINABLE_LAYERS = ["layer3", "layer4", "fc"] # Trains the deeper halves.
ANATCL_FOLD             = 0                          # Specific pre-trained fold for AnatCL.
ANATCL_DESCRIPTOR       = "global"                   # Specific feature descriptor type for AnatCL.
# BRAINIAC_UNFREEZE_MODE: Controls which blocks are unfrozen.
# - "linear_probe": Head only. Fastest, weakest. Good baseline.
# - "last3":        Unfreeze blocks 9-11 + norm. Recommended starting point.
# - "last6":        Unfreeze blocks 6-11 + norm. More capacity, needs more data.
# - "full":         Unfreeze everything.
BRAINIAC_UNFREEZE_MODE  = "last3"
# BRAINIAC_BACKBONE_LR: Separate LR for unfrozen backbone blocks (should be << head LR).
# Only used when BRAINIAC_UNFREEZE_MODE != "linear_probe".
BRAINIAC_BACKBONE_LR    = 5e-5

# ── Visualisation & Interpretability ──────────────────────────────────────────
# HEATMAP_ALPHA: Transparency of the Grad-CAM overlay when plotting (0.0 to 1.0).
HEATMAP_ALPHA    = 0.5
# HEATMAP_CMAP: The color palette for the saliency map. "jet" goes from Blue (low importance) to Red (high importance).
HEATMAP_CMAP     = "jet"
# SAVE_DPI: Image resolution for saved plots. 150-300 is standard for publications.
SAVE_DPI         = 150

# GRADCAM_LAYER: The specific layer in the neural network from which to extract
# the saliency gradients. Grad-CAM works best when extracting from the *final* # spatial convolutional layer before the image is flattened.
if CHOSEN_MODEL == "SFCN":        
    GRADCAM_LAYER = "features.5.2"
elif CHOSEN_MODEL == "ResNet18":  
    GRADCAM_LAYER = "layer4.1.conv2"
elif CHOSEN_MODEL == "ResNet50":  
    GRADCAM_LAYER = "layer4.2.conv2"
elif CHOSEN_MODEL == "DenseNet121":
    GRADCAM_LAYER = "features.denseblock4.denselayer16.layers.conv2"
elif CHOSEN_MODEL == "BrainIAC":
    GRADCAM_LAYER = None   # not used — attention rollout handles saliency
else:                            
    GRADCAM_LAYER = "backbone.encoder.layer4"

if TEST:
    EPOCHS = 1
    BATCH_SIZE = 2
    NUM_WORKERS = 0
    USE_SITE_HOLDOUT = False
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.2

# [1] Imports & Module-Level Definitions
import os
import sys
import shutil
from datetime import datetime
import multiprocessing

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

import monai
from monai.data import Dataset, CacheDataset, DataLoader
from monai.networks.nets import ResNet, DenseNet121
from monai.networks.nets.resnet import ResNetBlock
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    LoadImaged, NormalizeIntensityd, RandAffined,
    RandGaussianNoised, RandGaussianSmoothd, Resized,
)
from monai.visualize import GradCAM

try:
    import types
    import anatcl.models as _anatcl_models
    import anatcl.models.resnet3d as _anatcl_resnet3d

    class AgeEstimator(torch.nn.Module):
        def __init__(self, encoder, num_classes=1):
            super().__init__()
            self.encoder = encoder
            self.fc = torch.nn.Linear(512, num_classes)
        def forward(self, x):
            return self.fc(self.encoder(x))

    sys.modules['models'] = _anatcl_models

    _fake_estimators = types.ModuleType('models.estimators')
    for _k, _v in vars(_anatcl_resnet3d).items():
        if not _k.startswith('__'):
            setattr(_fake_estimators, _k, _v)
    _fake_estimators.AgeEstimator = AgeEstimator

    sys.modules['models.estimators']        = _fake_estimators
    sys.modules['anatcl.models.estimators'] = _fake_estimators

    from anatcl import AnatCL

except ImportError:
    print("  [WARNING] AnatCL not found.")
    AnatCL = None

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SFCN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            self._make_block(1, 32), self._make_block(32, 64), self._make_block(64, 128),
            self._make_block(128, 256), self._make_block(256, 256), self._make_block(256, 64, pool=False),
        )
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Conv3d(64, num_classes, kernel_size=1))

    def _make_block(self, in_ch, out_ch, pool=True):
        layers = [nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1), nn.BatchNorm3d(out_ch), nn.ReLU(inplace=False)]
        if pool: layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool3d(self.features(x), 1)
        return self.classifier(x).view(x.size(0), -1)

class BrainIACClassifier(nn.Module):
    """
    Wraps BrainIAC ViT-B backbone with a classification head.
    Backbone outputs 768-dim CLS token → head maps to num_classes.
    """
    def __init__(self, ckpt_path, num_classes=2, unfreeze_mode="linear_probe"):
        super().__init__()
        brainiac_src = os.path.abspath(
            os.path.join(os.path.dirname(__file__), 'BrainIAC-main', 'src')
        )
        if brainiac_src not in sys.path:
            sys.path.insert(0, brainiac_src)
        from model import ViTBackboneNet
        self.backbone = ViTBackboneNet(simclr_ckpt_path=ckpt_path)

        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        if unfreeze_mode == "full":
            for param in self.backbone.parameters():
                param.requires_grad = True
            print("INFO: BrainIAC backbone UNFROZEN (full fine-tune).")
        elif unfreeze_mode == "linear_probe":
            print("INFO: BrainIAC backbone FROZEN (linear probe).")
        elif unfreeze_mode in ("last3", "last6"):
            blocks_to_unfreeze = [9, 10, 11] if unfreeze_mode == "last3" else [6, 7, 8, 9, 10, 11]
            for name, param in self.backbone.named_parameters():
                if any(f"backbone.blocks.{i}" in name for i in blocks_to_unfreeze):
                    param.requires_grad = True
                if "backbone.norm" in name:
                    param.requires_grad = True
            print(f"INFO: BrainIAC partially unfrozen ({unfreeze_mode}: blocks {blocks_to_unfreeze} + norm).")

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)   # (B, 768)
        return self.head(features)    # (B, num_classes)

class BrainIACAttentionSaliency:
    """
    Attention Rollout for BrainIAC ViT-B.
    Captures all 12 attention layers, multiplies them sequentially,
    then upsamples the CLS-token attention row to full MRI volume size.
    """
    def __init__(self, model):
        self.model    = model
        self.attn_maps = []
        self.hooks    = []
        for name, module in self.model.named_modules():
            if module.__class__.__name__ == "SABlock":
                self.hooks.append(
                    module.register_forward_hook(
                        lambda m, i, o: self.attn_maps.append(
                            m.att_mat.detach().cpu() if hasattr(m, 'att_mat') else None
                        )
                    )
                )

    def _rollout(self, discard_ratio=0.9):
        valid = [a for a in self.attn_maps if a is not None]
        if not valid:
            return None
        result = torch.eye(valid[0].size(-1))
        for attn in valid:
            a = attn[0].mean(dim=0)
            flat = a.view(-1)
            threshold = flat.kthvalue(int(flat.size(0) * discard_ratio))[0]
            a = a.clone()
            a[a < threshold] = 0
            a = a + torch.eye(a.size(-1))
            a = a / (a.sum(dim=-1, keepdim=True) + 1e-8)
            result = torch.matmul(a, result)
        return result[0, 1:]  # CLS → patch tokens

    def generate(self, input_image):
        self.attn_maps = []
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_image)
        cls_attn = self._rollout()
        for h in self.hooks:
            h.remove()
        if cls_attn is None:
            print("  [WARNING] No attention maps captured for BrainIAC.")
            return np.zeros(input_image.shape[2:])

        n = cls_attn.shape[0]
        # Find closest perfect cube, trim if needed
        g = round(n ** (1/3))
        while g ** 3 > n:
            g -= 1
        g_cubed = g ** 3
        patch_map = cls_attn[:g_cubed].reshape(g, g, g).numpy()

        scale = tuple(t / p for t, p in zip(input_image.shape[2:], patch_map.shape))
        sal = zoom(patch_map, scale, order=1)
        sal = np.maximum(sal, 0)
        sal /= (sal.max() + 1e-8)
        return sal

def get_model(model_name, device):
    print(f"\nInitializing {model_name}...")

    if model_name == "ResNet18":
        model = ResNet(block=ResNetBlock, layers=[2, 2, 2, 2], block_inplanes=[64, 128, 256, 512],
                       spatial_dims=3, n_input_channels=1, num_classes=NUM_CLASSES).to(device)
        if os.path.exists(WEIGHT_PATH):
            checkpoint = torch.load(WEIGHT_PATH, map_location=device)
            pretrained_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items() 
                              if k.replace("module.", "") in model.state_dict() and "fc" not in k}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {len(new_state_dict)} MedicalNet weights.")
        for name, param in model.named_parameters():
            param.requires_grad = any(x in name for x in RESNET_TRAINABLE_LAYERS)
    
    elif model_name == "ResNet50":
        model = ResNet(block=ResNetBlock, layers=[3, 4, 6, 3], block_inplanes=[64, 128, 256, 512],
                       spatial_dims=3, n_input_channels=1, num_classes=NUM_CLASSES).to(device)
        if os.path.exists(WEIGHT_PATH):
            checkpoint = torch.load(WEIGHT_PATH, map_location=device)
            pretrained_dict = checkpoint.get('state_dict', checkpoint)
            new_state_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items() 
                              if k.replace("module.", "") in model.state_dict() and "fc" not in k}
            model.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded {len(new_state_dict)} MedicalNet weights.")
        for name, param in model.named_parameters():
            param.requires_grad = any(x in name for x in RESNET_TRAINABLE_LAYERS)

    elif model_name == "DenseNet121":
        model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=NUM_CLASSES).to(device)

    elif model_name == "AnatCL":
        model = AnatCL(descriptor=ANATCL_DESCRIPTOR, fold=ANATCL_FOLD, pretrained=True).to(device)
        # backbone.encoder outputs 512-dim, backbone.head is the contrastive projection head
        model.backbone.head = nn.Linear(512, NUM_CLASSES).to(device)        
        # Freeze all layers, then unfreeze the trainable ones
        for name, param in model.named_parameters():
            param.requires_grad = any(layer in name for layer in ANATCL_TRAINABLE_LAYERS)

    elif model_name == "SFCN":
        model = SFCN(num_classes=NUM_CLASSES).to(device)
        if os.path.exists(WEIGHT_PATH):
            model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device), strict=False)
        for name, param in model.named_parameters():
            param.requires_grad = not any(f"features.{i}" in name for i in SFCN_FROZEN_BLOCKS)
    
    elif model_name == "BrainIAC":
        if not os.path.exists(WEIGHT_PATH):
            raise FileNotFoundError(f"BrainIAC weights not found at {WEIGHT_PATH}")
        model = BrainIACClassifier(
            ckpt_path=WEIGHT_PATH,
            num_classes=NUM_CLASSES,
            unfreeze_mode=BRAINIAC_UNFREEZE_MODE
        ).to(device)
        print(f"Loaded BrainIAC from {WEIGHT_PATH}")
    
    return model

class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.hooks = []
        def backward_hook(module, grad_in, grad_out):
            if isinstance(module, nn.ReLU): return (torch.clamp(grad_in[0], min=0.0),)
        for module in self.model.modules():
            if isinstance(module, nn.ReLU): self.hooks.append(module.register_full_backward_hook(backward_hook))

    def generate(self, input_image, target_class):
        # ── Disable inplace ReLU to prevent view+inplace conflict ──
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

        self.model.eval()
        input_image = input_image.clone()  # Detach from any existing graph
        input_image.requires_grad = True
        output = self.model(input_image)
        self.model.zero_grad()
        target = torch.zeros_like(output)
        target[0, target_class] = 1
        output.backward(gradient=target)
        for hook in self.hooks: hook.remove()
        return input_image.grad.detach().cpu().numpy()[0, 0]

def get_anatomical_ranking(sub_id, binary_mask_3d):
    seg_path = os.path.join(BIDS_ROOT, "derivatives", "labels", f"sub-{sub_id}", f"{sub_id}_desc-synthseg_dseg.nii.gz")
    vol_path = os.path.join(BIDS_ROOT, "derivatives", "labels", f"sub-{sub_id}", f"{sub_id}_desc-synthseg_volumes.csv")
 
    if not os.path.exists(seg_path):
        print(f"  [WARNING] No SynthSeg derivative found for sub-{sub_id}. Skipping anatomical ranking.")
        return None
    
    CSV_COL_TO_LABEL_ID = {
        "left cerebral white matter":       2,  "left cerebral cortex":           3,
        "left lateral ventricle":           4,  "left inferior lateral ventricle": 5,
        "left cerebellum white matter":     7,  "left cerebellum cortex":          8,
        "left thalamus":                   10,  "left caudate":                   11,
        "left putamen":                    12,  "left pallidum":                  13,
        "left hippocampus":                17,  "left amygdala":                  18,
        "left accumbens area":             26,  "left ventral dc":                28,
        "right cerebral white matter":     41,  "right cerebral cortex":          42,
        "right lateral ventricle":         43,  "right inferior lateral ventricle":44,
        "right cerebellum white matter":   46,  "right cerebellum cortex":        47,
        "right thalamus":                  49,  "right caudate":                  50,
        "right putamen":                   51,  "right pallidum":                 52,
        "right hippocampus":               53,  "right amygdala":                 54,
        "right accumbens area":            58,  "right ventral dc":               60,
        "3rd ventricle":                   14,  "4th ventricle":                  15,
        "brain-stem":                      16,
    }

    region_volume_mm3 = {}  # label_id (int) → volume in mm³
    if os.path.exists(vol_path):
        try:
            vol_df = pd.read_csv(vol_path)
            # Drop completely empty rows (the blank separator rows in the CSV)
            vol_df = vol_df.dropna(how="all")
            # Find the data row: the one whose first column contains the subject string
            data_row = vol_df[vol_df.iloc[:, 0].astype(str).str.contains(sub_id, na=False)]
            if not data_row.empty:
                data_row = data_row.iloc[0]
                for col in vol_df.columns[1:]:          # skip the subject-ID column
                    col_key = col.strip().lower()
                    label_id = CSV_COL_TO_LABEL_ID.get(col_key)
                    if label_id is not None:
                        try:
                            region_volume_mm3[label_id] = float(data_row[col])
                        except (ValueError, TypeError):
                            pass
            else:
                print(f"  [WARNING] Subject {sub_id} not found in volume CSV. Volume normalisation unavailable.")
        except Exception as e:
            print(f"  [WARNING] Could not parse volume CSV for sub-{sub_id}: {e}")
    else:
        print(f"  [WARNING] No SynthSeg volume CSV found at {vol_path}. Volume normalisation unavailable.")
    
    label_map = {
        # Left Hemisphere
        2: "Left-Cerebral-White-Matter", 
        3: "Left-Cerebral-Cortex", 
        4: "Left-Lateral-Ventricle", 
        5: "Left-Inferior-Lateral-Ventricle",
        7: "Left-Cerebellum-White-Matter",
        8: "Left-Cerebellum-Cortex",
        10: "Left-Thalamus", 
        11: "Left-Caudate", 
        12: "Left-Putamen", 
        13: "Left-Pallidum", 
        17: "Left-Hippocampus", 
        18: "Left-Amygdala",
        26: "Left-Accumbens-Area",
        28: "Left-Ventral-DC",
        
        # Right Hemisphere
        41: "Right-Cerebral-White-Matter", 
        42: "Right-Cerebral-Cortex", 
        43: "Right-Lateral-Ventricle",
        44: "Right-Inferior-Lateral-Ventricle",
        46: "Right-Cerebellum-White-Matter",
        47: "Right-Cerebellum-Cortex",
        49: "Right-Thalamus", 
        50: "Right-Caudate", 
        51: "Right-Putamen", 
        52: "Right-Pallidum", 
        53: "Right-Hippocampus", 
        54: "Right-Amygdala",
        58: "Right-Accumbens-Area",
        60: "Right-Ventral-DC",

        # Midline / Central Structures
        14: "3rd-Ventricle",
        15: "4th-Ventricle",
        16: "Brain-Stem"
    }

    # ── Load segmentation & compute saliency overlap ──────────────────────────
    seg_obj  = nib.load(seg_path)
    seg_data = seg_obj.get_fdata()

    native_voxel_dims       = np.abs(np.diag(seg_obj.affine)[:3])
    native_voxel_vol_mm3    = float(np.prod(native_voxel_dims))

    scale_factors           = np.array(binary_mask_3d.shape) / np.array(seg_data.shape)
    resized_seg             = zoom(seg_data, scale_factors, order=0)
    resampled_voxel_vol_mm3 = float(np.prod(native_voxel_dims / scale_factors))

    important_voxels        = resized_seg[binary_mask_3d > 0]
    
    unique_labels, counts = np.unique(important_voxels, return_counts=True)

    ranking = []
    total_salient  = np.sum(counts)

    for lbl, cnt in zip(unique_labels, counts):
        if lbl == 0:
            continue
        lbl_int  = int(lbl)
        name     = label_map.get(lbl_int, f"Region-{lbl_int}")
        raw_pct  = (cnt / total_salient) * 100

        # Convert salient voxel count to mm³ using the segmentation voxel size
        salient_mm3 = cnt * resampled_voxel_vol_mm3

        # Volume-normalised importance: what fraction of the region's volume
        # was flagged as salient?  Expressed as a percentage (0–100%).
        region_mm3  = region_volume_mm3.get(lbl_int)
        if region_mm3 and region_mm3 > 0:
            vol_norm_pct = (salient_mm3 / region_mm3) * 100
        else:
            vol_norm_pct = None   # CSV missing for this region → don't fabricate a number

        ranking.append({
            "Region":                name,
            "SaliencyVoxels":        int(cnt),
            "SaliencyVolume_mm3":    round(salient_mm3, 2),
            "RegionVolume_mm3":      round(region_mm3, 2) if region_mm3 else None,
            "RawImportance%":        round(raw_pct, 2),
            "VolumeNormImportance%": round(vol_norm_pct, 2) if vol_norm_pct is not None else None,
        })

    if not ranking:   # ADD THIS GUARD
        print(f"  [WARNING] No labelled regions overlapped with saliency mask for sub-{sub_id}.")
        print(f"  Debug: seg shape={seg_data.shape}, mask shape={binary_mask_3d.shape}, "
                f"mask nonzero={np.sum(binary_mask_3d)}, seg unique={np.unique(resized_seg[:5,:5,:5])}")
        return None

    df = pd.DataFrame(ranking)

    # Raw ranking: sorted by what fraction of total saliency fell in each region
    df_raw = (df[["Region", "SaliencyVoxels", "SaliencyVolume_mm3", "RawImportance%"]]
                .sort_values(by="RawImportance%", ascending=False)
                .reset_index(drop=True))

    # Volume-normalised ranking: sorted by what fraction of each region was salient
    # Rows where the CSV volume was missing are pushed to the bottom
    df_norm = (df[["Region", "SaliencyVoxels", "SaliencyVolume_mm3",
                   "RegionVolume_mm3", "VolumeNormImportance%"]]
                 .sort_values(by="VolumeNormImportance%", ascending=False, na_position="last")
                 .reset_index(drop=True))

    return df_raw, df_norm

# Helper function to get predictions for a given dataloader
def get_predictions(model, loader, device):
    t_true, t_probs, t_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            t_true.extend(batch["label"].numpy())
            t_probs.extend(torch.softmax(model(batch["image"].to(device)), dim=1)[:, 1].cpu().numpy())
            t_ids.extend(batch["sub_id"])
    return np.array(t_true), np.array(t_probs), np.array(t_ids)

# Helper function to plot and save prediction histograms
def plot_output_histogram(true_labels, probs, save_path, title):
    plt.figure(figsize=(8, 5), facecolor="white")
    plt.hist(probs[true_labels == 0], bins=20, alpha=0.6, label='Control (0)', color='blue')
    plt.hist(probs[true_labels == 1], bins=20, alpha=0.6, label='ADHD (1)', color='red')
    plt.xlabel('Predicted Probability (Class 1)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=SAVE_DPI)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION BLOCK
# ══════════════════════════════════════════════════════════════════════════════

def main():

    pin_memory = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # create run directory immediately so checkpoints go there
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(RUNS_DIR, f"{run_timestamp}_{CHOSEN_MODEL}")
    os.makedirs(run_dir, exist_ok=True)

    # [2] Data Parsing & Transforms

    participants = pd.read_csv(os.path.join(BIDS_ROOT, "participants.tsv"), sep="\t")
    if TEST:
        print("TEST MODE: Using only the first 10 entries from participants.tsv")
        participants = participants.head(10)

    # Skip entries with missing label values
    participants = participants[participants[LABEL_COLUMN].notnull()]

    images, labels, sites, sub_ids = [], [], [], []
    for _, row in participants.iterrows():
        sub_id = str(row["participant_id"])
        label  = row[LABEL_COLUMN]
        site   = row["site"] if "site" in row else "Unknown"
        
        img_path = os.path.join(BIDS_ROOT, f"sub-{sub_id}", "anat", f"{sub_id}_T1w.nii.gz")
        if os.path.isfile(img_path):
            images.append(img_path)
            labels.append(label)
            sites.append(site)
            sub_ids.append(sub_id)

    images_array  = np.array(images)
    labels_array  = torch.tensor(labels).long().numpy()
    sites_array   = np.array(sites)
    sub_ids_array = np.array(sub_ids)

    train_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=TARGET_SIZE, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandAffined(keys=["image"], prob=AUG_AFFINE_PROB, rotate_range=AUG_AFFINE_ROTATE, translate_range=AUG_AFFINE_TRANSLATE),
        RandGaussianNoised(keys=["image"], prob=AUG_NOISE_PROB, mean=0.0, std=AUG_NOISE_STD),
        RandGaussianSmoothd(keys=["image"], prob=AUG_SMOOTH_PROB, sigma_x=AUG_SMOOTH_SIGMA, sigma_y=AUG_SMOOTH_SIGMA, sigma_z=AUG_SMOOTH_SIGMA),
        EnsureTyped(keys=["image"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Resized(keys=["image"], spatial_size=TARGET_SIZE, mode="trilinear"),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"]),
    ])

    # [3] Dataloaders & Splitting 
    if USE_SITE_HOLDOUT:
        print(f"\n[SITE HOLDOUT ENABLED] Holding out site '{HOLDOUT_SITE}' as the Test Set.")
        test_mask = sites_array == HOLDOUT_SITE
        
        test_images, test_labels, test_ids = images_array[test_mask], labels_array[test_mask], sub_ids_array[test_mask]
        train_val_images, train_val_labels, train_val_ids = images_array[~test_mask], labels_array[~test_mask], sub_ids_array[~test_mask]
        
        train_images, val_images, train_labels, val_labels, train_ids, val_ids = train_test_split(
            train_val_images, train_val_labels, train_val_ids, test_size=VAL_SPLIT, stratify=train_val_labels, random_state=RANDOM_STATE
        )
    else:
        print("\n[RANDOM STRATIFIED SPLIT ENABLED]")
        train_val_images, test_images, train_val_labels, test_labels, train_val_ids, test_ids = train_test_split(
            images_array, labels_array, sub_ids_array, test_size=TEST_SPLIT, stratify=labels_array, random_state=RANDOM_STATE
        )
        relative_val_split = VAL_SPLIT / (1.0 - TEST_SPLIT)
        train_images, val_images, train_labels, val_labels, train_ids, val_ids = train_test_split(
            train_val_images, train_val_labels, train_val_ids, test_size=relative_val_split, stratify=train_val_labels, random_state=RANDOM_STATE
        )

    train_data_dicts = [{"image": img, "label": torch.tensor(lbl).long(), "sub_id": sid} for img, lbl, sid in zip(train_images, train_labels, train_ids)]
    val_data_dicts   = [{"image": img, "label": torch.tensor(lbl).long(), "sub_id": sid} for img, lbl, sid in zip(val_images, val_labels, val_ids)]
    test_data_dicts  = [{"image": img, "label": torch.tensor(lbl).long(), "sub_id": sid} for img, lbl, sid in zip(test_images, test_labels, test_ids)]

    class_counts   = np.bincount(train_labels)
    class_weights  = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler        = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    if USE_CACHE_DATASET:
        train_ds = CacheDataset(data=train_data_dicts, transform=train_transforms, cache_rate=1.0)
        val_ds   = CacheDataset(data=val_data_dicts, transform=val_transforms, cache_rate=1.0)
        test_ds  = CacheDataset(data=test_data_dicts, transform=val_transforms, cache_rate=1.0)
    else:
        train_ds = Dataset(data=train_data_dicts, transform=train_transforms)
        val_ds   = Dataset(data=val_data_dicts, transform=val_transforms)
        test_ds  = Dataset(data=test_data_dicts, transform=val_transforms)

    # Note: Setting shuffle=False for train evaluate later is needed to keep IDs aligned, 
    # but sampler handles training order. We'll create a sequential eval loader for train later.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin_memory)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    # Eval loader for the training set (no shuffling/sampler to easily map IDs)
    train_eval_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin_memory)

    # [5] Training
    model        = get_model(CHOSEN_MODEL, device)
    
    if USE_CLASS_WEIGHTS:
        class_counts_all = np.bincount(labels_array)
        cw = torch.tensor(1.0 / class_counts_all, dtype=torch.float).to(device)
        cw = cw / cw.sum()  # normalise so weights sum to 1
        loss_fn = nn.CrossEntropyLoss(weight=cw)
        print(f"Using class weights: {cw.cpu().numpy()} (Control={cw[0]:.4f}, ADHD={cw[1]:.4f})")
    else:
        loss_fn = nn.CrossEntropyLoss() 
           
    if CHOSEN_MODEL == "BrainIAC" and (BRAINIAC_UNFREEZE_MODE != "linear_probe"):
        optimizer = torch.optim.AdamW([
            {"params": model.head.parameters(),
            "lr": LEARNING_RATE},
            {"params": [p for n, p in model.backbone.named_parameters() if p.requires_grad],
            "lr": BRAINIAC_BACKBONE_LR},
        ], weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    print("[DEBUG]")
    for name, _ in model.named_modules():
        print(name)

    if SCHEDULER_TYPE == "Plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=SCHEDULER_T_MAX)

    epoch_metrics = []   
    best_val_auc  = -1
    val_acc = -1
    epochs_no_improve = 0  

    print(f"\nTraining {CHOSEN_MODEL} for up to {EPOCHS} epochs")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, lbls = batch["image"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inputs), lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for batch in val_loader:
                v_inputs, v_labels = batch["image"].to(device), batch["label"].to(device)
                y_true.extend(v_labels.cpu().numpy())
                y_probs.extend(torch.softmax(model(v_inputs), dim=1)[:, 1].cpu().numpy())

        try: auc = roc_auc_score(y_true, y_probs)
        except ValueError: auc = 0.5
        
        acc = np.mean((np.array(y_probs) > 0.5) == np.array(y_true))
        avg_loss = train_loss / len(train_loader)
        
        scheduler.step(auc) if SCHEDULER_TYPE == "Plateau" else scheduler.step()

        print(f"Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Val AUC: {auc:.4f} | Val Acc: {acc:.4f}")
        epoch_metrics.append({"epoch": epoch+1, "train_loss": avg_loss, "val_auc": auc, "val_acc": acc})

        if auc > best_val_auc:
            best_val_auc = auc
            val_acc = acc
            epochs_no_improve = 0
            # save best checkpoint directly into the run directory
            torch.save(model.state_dict(), os.path.join(run_dir, f"best_{CHOSEN_MODEL}.pth"))
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # [6] Guided Grad-CAM Saliency & Anatomical Lookup
    model.load_state_dict(torch.load(os.path.join(run_dir, f"best_{CHOSEN_MODEL}.pth")))
    model.eval()

    test_batch  = next(iter(test_loader))
    img         = test_batch["image"][0:1].to(device)
    label       = test_batch["label"][0:1].to(device)
    sample_id   = test_batch["sub_id"][0] 
    
    cam = GradCAM(nn_module=model, target_layers=GRADCAM_LAYER)
    if CHOSEN_MODEL == "BrainIAC":
        # ── Attention Rollout saliency ──────────────────────────────────────
        saliency_method = "Attention Rollout"
        extractor       = BrainIACAttentionSaliency(model)
        guided_gradcam  = extractor.generate(img)
    else:
        # ── Guided Grad-CAM saliency (CNN models) ───────────────────────────
        saliency_method = "Guided Grad-CAM"
        gradcam_result  = cam(x=img, class_idx=label.item())[0, 0].cpu().numpy()
        guided_grads    = GuidedBackprop(model).generate(img, label.item())
        guided_gradcam  = np.maximum(guided_grads * gradcam_result, 0)
        guided_gradcam /= (np.max(guided_gradcam) + 1e-8)

    valid_voxels   = guided_gradcam[guided_gradcam > 0]
    q50            = np.median(valid_voxels) if len(valid_voxels) > 0 else 0.0
    binary_mask_3d = (guided_gradcam > q50).astype(int)

    print(f"\n--- Anatomical Importance for Subject {sample_id} [{saliency_method}] ---")
    result = get_anatomical_ranking(sample_id, binary_mask_3d)
    if result is not None:
        importance_df_raw, importance_df_norm = result
        print(f"\n--- Raw Importance (sub-{sample_id}) ---")
        print(importance_df_raw.head(10))
        print(f"\n--- Volume-Normalised Importance (sub-{sample_id}) ---")
        print(importance_df_norm.head(10))
    else:
        importance_df_raw, importance_df_norm = None, None

    # [7] Results, Plotting & Final Testing
    # run_dir already created at the start of main; timestamp computed earlier

    # Save Hyperparameters
    hyperparams_dict = {
        # Run metadata
        "run_timestamp": run_timestamp,
        "run_dir": run_dir,
        # Paths
        "BIDS_ROOT": BIDS_ROOT,
        "RUNS_DIR": RUNS_DIR,
        "WEIGHT_PATH": WEIGHT_PATH,

        # Model
        "CHOSEN_MODEL": CHOSEN_MODEL,
        "NUM_CLASSES": NUM_CLASSES,

        # Data & Splitting
        "LABEL_COLUMN": LABEL_COLUMN,
        "TARGET_SIZE": TARGET_SIZE,
        "VAL_SPLIT": VAL_SPLIT,
        "TEST_SPLIT": TEST_SPLIT,
        "RANDOM_STATE": RANDOM_STATE,
        "USE_SITE_HOLDOUT": USE_SITE_HOLDOUT,
        "HOLDOUT_SITE": HOLDOUT_SITE,

        # Advanced Training Options
        "USE_CACHE_DATASET": USE_CACHE_DATASET,
        "FIND_OPTIMAL_THRESHOLD": FIND_OPTIMAL_THRESHOLD,
        "EARLY_STOPPING_PATIENCE": EARLY_STOPPING_PATIENCE,
        "SCHEDULER_TYPE": SCHEDULER_TYPE,
        "SCHEDULER_PATIENCE": SCHEDULER_PATIENCE,
        "SCHEDULER_FACTOR": SCHEDULER_FACTOR,
        "USE_CLASS_WEIGHTS": USE_CLASS_WEIGHTS,

        # Augmentation
        "AUG_AFFINE_PROB": AUG_AFFINE_PROB,
        "AUG_AFFINE_ROTATE": AUG_AFFINE_ROTATE,
        "AUG_AFFINE_TRANSLATE": AUG_AFFINE_TRANSLATE,
        "AUG_NOISE_PROB": AUG_NOISE_PROB,
        "AUG_NOISE_STD": AUG_NOISE_STD,
        "AUG_SMOOTH_PROB": AUG_SMOOTH_PROB,
        "AUG_SMOOTH_SIGMA": AUG_SMOOTH_SIGMA,

        # Training
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,
        "LEARNING_RATE": LEARNING_RATE,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "SCHEDULER_T_MAX": SCHEDULER_T_MAX,

        # Fine-tuning Options
        "SFCN_FROZEN_BLOCKS": SFCN_FROZEN_BLOCKS,
        "ANATCL_TRAINABLE_LAYERS": ANATCL_TRAINABLE_LAYERS,
        "RESNET_TRAINABLE_LAYERS": RESNET_TRAINABLE_LAYERS,
        "ANATCL_FOLD": ANATCL_FOLD,
        "ANATCL_DESCRIPTOR": ANATCL_DESCRIPTOR,
        "BRAINIAC_UNFREEZE_MODE": BRAINIAC_UNFREEZE_MODE,
        "BRAINIAC_BACKBONE_LR": BRAINIAC_BACKBONE_LR,

        # Visualisation & Interpretability
        "HEATMAP_ALPHA": HEATMAP_ALPHA,
        "HEATMAP_CMAP": HEATMAP_CMAP,
        "SAVE_DPI": SAVE_DPI,
        "GRADCAM_LAYER": GRADCAM_LAYER,
    }
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams_dict, f, indent=4)

    if importance_df_raw is not None:
        importance_df_raw.to_csv(os.path.join(run_dir,  f"sub-{sample_id}_anatomical_importance_raw.csv"),  index=False)
        importance_df_norm.to_csv(os.path.join(run_dir, f"sub-{sample_id}_anatomical_importance_norm.csv"), index=False)

    # Plot Training Curves
    df_metrics = pd.DataFrame(epoch_metrics)
    fig_curves, ax1 = plt.subplots(figsize=(10, 5), facecolor="white")
    color_loss = 'tab:blue'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Average Training Loss', color=color_loss)
    ax1.plot(df_metrics["epoch"], df_metrics["train_loss"], color=color_loss, marker="o", label='Avg Training Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # plot both validation accuracy and AUC on the secondary axis
    ax2 = ax1.twinx()
    color_acc = 'tab:orange'
    ax2.set_ylabel('Validation Accuracy / AUC', color=color_acc)
    ax2.plot(df_metrics["epoch"], df_metrics["val_acc"], color=color_acc, marker="s", label='Val Accuracy')
    # add auc as a second line sharing the same axis
    color_auc = 'tab:green'
    ax2.plot(df_metrics["epoch"], df_metrics.get("val_auc", []), color=color_auc, marker="^", label='Val AUC')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    # combine legends from both axes
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig_curves.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

    plt.title(f"Training Curves: {CHOSEN_MODEL}")
    fig_curves.tight_layout()
    fig_curves.savefig(os.path.join(run_dir, "loss_vs_accuracy.png"), dpi=SAVE_DPI)
    plt.close(fig_curves)

    # Save Saliency Heatmap
    depth_slice = img.shape[2] // 2
    orig_slice = img[0, 0, depth_slice, ...].detach().cpu().numpy()
    heatmap_2d = guided_gradcam[depth_slice, ...]
    mask_2d = binary_mask_3d[depth_slice, ...].astype(float)
    mask_2d[mask_2d == 0] = np.nan

    fig_sal, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="white")
    axes[0].imshow(orig_slice, cmap="gray")
    axes[0].set_title(f"Original T1 (sub-{sample_id})")
    axes[1].imshow(orig_slice, cmap="gray")
    axes[1].imshow(heatmap_2d, cmap=HEATMAP_CMAP, alpha=HEATMAP_ALPHA)
    axes[1].set_title(f"{saliency_method} [{CHOSEN_MODEL}]")
    axes[2].imshow(orig_slice, cmap="gray")
    axes[2].imshow(mask_2d, cmap="Reds", alpha=0.7)
    axes[2].set_title(f"Predictive Mask (q > {q50:.3f})")
    for ax in axes: ax.axis("off")
    fig_sal.tight_layout()
    fig_sal.savefig(os.path.join(run_dir, f"saliency_sub-{sample_id}.png"), dpi=SAVE_DPI)
    plt.close(fig_sal)

    # --- Full Data Evaluation Block ---
    print("\nExtracting final predictions for Train, Val, and Test sets...")
    
    # 1. Evaluate Train Set
    train_true, train_probs, train_ids_arr = get_predictions(model, train_eval_loader, device)
    train_preds = (train_probs > 0.5).astype(int)
    pd.DataFrame({"sub_id": train_ids_arr, "true_label": train_true, "pred_prob": train_probs, "pred_label": train_preds}).to_csv(os.path.join(run_dir, "train_predictions.csv"), index=False)
    plot_output_histogram(train_true, train_probs, os.path.join(run_dir, "train_histogram.png"), "Training Set Predictions")

    # 2. Evaluate Val Set
    val_true, val_probs, val_ids_arr = get_predictions(model, val_loader, device)
    
    if FIND_OPTIMAL_THRESHOLD:
        fpr, tpr, thresholds = roc_curve(val_true, val_probs)
        youden_j = tpr - fpr
        optimal_threshold = thresholds[np.argmax(youden_j)]
        print(f"Optimal Threshold found via Youden's J: {optimal_threshold:.4f}")
    else:
        optimal_threshold = 0.5
        print(f"Using default threshold: {optimal_threshold}")

    val_preds = (val_probs > optimal_threshold).astype(int)
    pd.DataFrame({"sub_id": val_ids_arr, "true_label": val_true, "pred_prob": val_probs, "pred_label": val_preds}).to_csv(os.path.join(run_dir, "val_predictions.csv"), index=False)
    plot_output_histogram(val_true, val_probs, os.path.join(run_dir, "val_histogram.png"), "Validation Set Predictions")

    # 3. Evaluate Test Set
    t_true, t_probs, t_ids_arr = get_predictions(model, test_loader, device)
    
    # Apply the threshold we just calculated (or defaulted to 0.5)
    t_preds = (t_probs > optimal_threshold).astype(int)
    test_auc = roc_auc_score(t_true, t_probs)
    test_acc = np.mean(t_preds == t_true)
    
    pd.DataFrame({"sub_id": t_ids_arr, "true_label": t_true, "pred_prob": t_probs, "pred_label": t_preds}).to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)
    plot_output_histogram(t_true, t_probs, os.path.join(run_dir, "test_histogram.png"), "Test Set Predictions")

    # ══════════════════════════════════════════════════════════════════════════════
    # Group Anatomical Analysis — all four outcome types (TP, TN, FP, FN)
    # Runs over both Validation and Test splits
    # ══════════════════════════════════════════════════════════════════════════════

    def _outcome_label(true, pred):
        if   true == 1 and pred == 1: return "True Positive"
        elif true == 0 and pred == 0: return "True Negative"
        elif true == 0 and pred == 1: return "False Positive"
        else:                         return "False Negative"

    def run_anatomical_analysis(split_name, ds, ids_arr, true_arr, preds_arr):
        """
        Run saliency → anatomy pipeline for every subject in a split.
        Saves detailed_anatomy_by_outcome.csv and two summary CSVs
        (raw / norm) into a split-named subfolder of run_dir.
        """
        print(f"\nRunning Group Anatomical Analysis on {split_name} set (TP, TN, FP, FN)...")
        split_dir = os.path.join(run_dir, f"anatomy_{split_name.lower()}")
        os.makedirs(split_dir, exist_ok=True)

        group_rankings = []

        for s_id, s_true, s_pred in zip(ids_arr, true_arr, preds_arr):
            outcome = _outcome_label(int(s_true), int(s_pred))

            idx    = np.where(ids_arr == s_id)[0][0]
            s_data = ds[idx]
            s_img  = s_data["image"].unsqueeze(0).to(device)

            # Saliency explains the model's actual prediction (correct or not)
            if CHOSEN_MODEL == "BrainIAC":
                s_extractor      = BrainIACAttentionSaliency(model)
                s_guided_gradcam = s_extractor.generate(s_img)
            else:
                s_gradcam_result = cam(x=s_img, class_idx=s_pred)[0, 0].cpu().numpy()
                s_guided_grads   = GuidedBackprop(model).generate(s_img, s_pred)
                s_guided_gradcam = np.maximum(s_guided_grads * s_gradcam_result, 0)
                s_guided_gradcam /= (np.max(s_guided_gradcam) + 1e-8)

            s_valid_voxels = s_guided_gradcam[s_guided_gradcam > 0]
            s_q50          = np.median(s_valid_voxels) if len(s_valid_voxels) > 0 else 0.0
            s_binary_mask  = (s_guided_gradcam > s_q50).astype(int)

            s_result = get_anatomical_ranking(s_id, s_binary_mask)
            if s_result is not None:
                s_df_raw, s_df_norm = s_result
                for s_df, tag in [(s_df_raw, "raw"), (s_df_norm, "norm")]:
                    s_df["sub_id"]  = s_id
                    s_df["outcome"] = outcome
                    s_df["ranking"] = tag
                group_rankings.append(pd.concat([s_df_raw, s_df_norm]))

        if not group_rankings:
            print(f"  -> No anatomical results for {split_name}. Check SynthSeg derivatives.")
            return

        full_df = pd.concat(group_rankings, ignore_index=True)
        full_df.to_csv(os.path.join(split_dir, "detailed_anatomy_by_outcome.csv"), index=False)

        for col, tag in [("RawImportance%", "raw"), ("VolumeNormImportance%", "norm")]:
            subset = full_df[full_df["ranking"] == tag].copy()
            if subset.empty or col not in subset.columns:
                continue
            summary = (subset.groupby(["outcome", "Region"])[col]
                             .mean()
                             .reset_index()
                             .sort_values(by=["outcome", col], ascending=[True, False]))
            summary.to_csv(os.path.join(split_dir, f"group_anatomical_summary_{tag}.csv"), index=False)

        # Console: top-5 regions per outcome (norm score)
        norm_subset = full_df[full_df["ranking"] == "norm"].copy()
        if not norm_subset.empty:
            print(f"\n  Top-5 regions by VolumeNormImportance% per outcome [{split_name}]:")
            for outcome, grp in norm_subset.groupby("outcome"):
                top5 = (grp.groupby("Region")["VolumeNormImportance%"]
                           .mean()
                           .nlargest(5)
                           .reset_index())
                print(f"\n  [{outcome}]")
                print(top5.to_string(index=False))

        print(f"  -> Saved {split_name} anatomical analysis to {split_dir}")

    # ── Run for both splits ────────────────────────────────────────────────────
    run_anatomical_analysis(
        split_name = "Val",
        ds         = val_ds,
        ids_arr    = val_ids_arr,
        true_arr   = val_true,
        preds_arr  = val_preds,
    )
    run_anatomical_analysis(
        split_name = "Test",
        ds         = test_ds,
        ids_arr    = t_ids_arr,
        true_arr   = t_true,
        preds_arr  = t_preds,
    )

    # Calculate Confusion Matrix & Metrics
    cm = confusion_matrix(t_true, t_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if len(cm.ravel()) == 4 else (0, 0, 0, 0) # Fallback if only 1 class is predicted
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Save Confusion Matrix Plot
    fig_cm, ax_cm = plt.subplots(figsize=(6, 6), facecolor="white")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Control (0)", "ADHD (1)"])
    disp.plot(cmap="Blues", ax=ax_cm, colorbar=False)
    plt.title("Test Set Confusion Matrix")
    fig_cm.savefig(os.path.join(run_dir, "confusion_matrix.png"), dpi=SAVE_DPI)
    plt.close(fig_cm)

    # Save Final Metrics CSV
    metrics_dict = {
        "Val AUC": best_val_auc,
        "Val Accuracy": val_acc,
        "Test AUC": test_auc,
        "Test Accuracy": test_acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "ntrain": len(train_ds),
        "nval": len(val_ds),
        "ntest": len(test_ds)
    }
    pd.DataFrame([metrics_dict]).to_csv(os.path.join(run_dir, "metrics.csv"), index=False)

    # checkpoint already saved inside run_dir during training; no copy needed
    df_metrics.to_csv(os.path.join(run_dir, "epoch_metrics.csv"), index=False)

    print(f"\nTesting Completed! Test AUC: {test_auc:.4f} | Test Acc: {test_acc:.4f}")
    print(f"Results saved to: {run_dir}")

# ══════════════════════════════════════════════════════════════════════════════
# MULTIPROCESSING ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()