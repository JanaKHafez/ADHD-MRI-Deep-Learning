import os
import pandas as pd
import subprocess
import tensorflow as tf
from pathlib import Path

# ── HARDWARE DIAGNOSTICS ──────────────────────────────────────────────────
print("\n" + "="*50)
print("🔍 HARDWARE CHECK")

# Detect logical CPU cores to maximize performance in CPU mode
num_threads = os.cpu_count() or 4
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print(f"✅ GPU DETECTED: {gpus}")
    print("🚀 Status: Processing will be fast (approx 20-40s / subject)")
    hardware_flag = [] 
else:
    print("⚠️  NO GPU DETECTED (Num GPUs: 0)")
    print(f"🐢 Status: Running on CPU using {num_threads} threads.")
    print("📝 Note: Approx 2-4 minutes per subject.")
    hardware_flag = ["--cpu"]

# Set environment variable to reduce TensorFlow logging noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
print("="*50 + "\n")

# ── CONFIGURATION ──────────────────────────────────────────────────────────
# 1. Paths to your project and venv
BIDS_ROOT       = r"C:\Users\janak\GitHub\ADHD-MRI-Deep-Learning\ADHD_BIDS"
SYNTHSEG_REPO   = r"C:\Users\janak\GitHub\ADHD-MRI-Deep-Learning\src\SynthSeg"
# Ensure this points to the python.exe in your active environment
PYTHON_EXE      = r"C:\Users\janak\GitHub\ADHD-MRI-Deep-Learning\Segment_VENV\Scripts\python.exe"

# 2. Path to the SynthSeg script inside the cloned repo
SYNTHSEG_SCRIPT = os.path.join(SYNTHSEG_REPO, "scripts", "commands", "SynthSeg_predict.py")

# 3. Path to your model file
MODEL_PATH      = os.path.join(SYNTHSEG_REPO, "models", "synthseg_1.0.h5")

# ── DIRECTORY SETUP ────────────────────────────────────────────────────────
DERIVATIVES_DIR = os.path.join(BIDS_ROOT, "derivatives", "labels")
os.makedirs(DERIVATIVES_DIR, exist_ok=True)

# ── PRE-FLIGHT CHECK ───────────────────────────────────────────────────────
if not os.path.exists(SYNTHSEG_SCRIPT):
    print(f"ERROR: Could not find SynthSeg script at: {SYNTHSEG_SCRIPT}")
    exit()

if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at: {MODEL_PATH}")
    print("Please ensure 'synthseg_1.0.h5' is inside the SynthSeg/models folder.")
    exit()

# ── LOAD DATA ──────────────────────────────────────────────────────────────
participants_path = os.path.join(BIDS_ROOT, "participants.tsv")
if not os.path.exists(participants_path):
    print(f"ERROR: Participants file not found at: {participants_path}")
    exit()

df = pd.read_csv(participants_path, sep="\t")
print(f"Found {len(df)} participants. Starting segmentation...\n")

# ── PROCESSING LOOP ────────────────────────────────────────────────────────
for idx, row in df.iterrows():
    sub_id = str(row['participant_id'])
    
    # Input path (assuming BIDS format: sub-ID/anat/sub-ID_T1w.nii.gz)
    input_img = os.path.join(BIDS_ROOT, f"sub-{sub_id}", "anat", f"{sub_id}_T1w.nii.gz")
    
    # Output paths
    sub_output_dir = os.path.join(DERIVATIVES_DIR, f"sub-{sub_id}")
    os.makedirs(sub_output_dir, exist_ok=True)
    
    output_seg = os.path.join(sub_output_dir, f"{sub_id}_desc-synthseg_dseg.nii.gz")
    output_vol = os.path.join(sub_output_dir, f"{sub_id}_desc-synthseg_volumes.csv")

    # Skip if input doesn't exist
    if not os.path.exists(input_img):
        print(f"  [MISSING] Input not found for sub-{sub_id}")
        continue

    # Skip if already processed
    if os.path.exists(output_seg) and os.path.exists(output_vol):
        print(f"  [SKIP] sub-{sub_id} already segmented.")
        continue

    print(f"  [RUN] Segmenting sub-{sub_id}...")
    
    # Build command
    cmd = [
        PYTHON_EXE,
        SYNTHSEG_SCRIPT,
        "--i", input_img,
        "--o", output_seg,
        "--vol", output_vol,
        "--threads", str(num_threads),
        "--v1" 
    ] + hardware_flag

    try:
        # Run subprocess
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True, 
            cwd=SYNTHSEG_REPO
        )
        print(f"  [DONE] Saved to {output_seg}")
        
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] Failed to process sub-{sub_id}")
        # Print only the last 500 characters of error to avoid terminal flooding
        error_msg = e.stderr if e.stderr else "Unknown error"
        print(f"  Reason: {error_msg[-500:]}")

    print(f"    Completed {idx + 1}/{len(df)} subjects.")

print("\n--- All subjects finished ---")