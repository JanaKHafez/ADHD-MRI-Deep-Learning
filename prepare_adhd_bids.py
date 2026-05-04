#!/usr/bin/env python3
"""
prepare_adhd_bids.py
Converts ADHD200 preprocessed data to BIDS-compliant format.
Creates a participants.tsv file with all subject metadata.
"""

import os
import pandas as pd
import shutil
from pathlib import Path

# Configuration
ADHD_ROOT = "ADHD_BIDS"
OUTPUT_ROOT = "ADHD_BIDS"

def merge_phenotypic_data():
    """
    Merge all site-specific phenotypic CSVs into a unified participants.tsv
    """
    print("[1] Merging phenotypic data from all sites...")
    
    all_data = []
    
    # Find all phenotypic CSV files in ADHD_ROOT
    pheno_files = [f for f in os.listdir(ADHD_ROOT) 
                   if f.endswith("_phenotypic.csv") and "TestRelease" not in f]
    
    # Extract site names from filenames
    sites = [f.replace("_phenotypic.csv", "") for f in pheno_files]
    
    print(f"Found {len(sites)} sites: {sites}\n")
    
    for site, pheno_file in zip(sites, pheno_files):
        pheno_path = os.path.join(ADHD_ROOT, pheno_file)
        print(f"  ✓ Reading {site}: {pheno_file}")
        
        try:
            df = pd.read_csv(pheno_path)
            df['site'] = site
            all_data.append(df)
        except Exception as e:
            print(f"    ERROR reading {pheno_path}: {e}")
            continue
    
    # Merge all DataFrames
    if not all_data:
        raise ValueError("No phenotypic data found!")
    merged_df = pd.concat(all_data, ignore_index=True, sort=False)
    
    print(f"\n✓ Merged {len(merged_df)} total records from {len(all_data)} sites")
    print(f"  Columns: {list(merged_df.columns)}\n")
    
    return merged_df

def process_subjects(merged_df):
    """
    Process subjects and create BIDS-compliant folder structure
    """
    print("[2] Processing subjects and creating BIDS structure...")
    
    # Create output directory if needed
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Standardize participant IDs
    if 'participant_id' not in merged_df.columns:
        if 'ScanDir ID' in merged_df.columns:
            merged_df['participant_id'] = merged_df['ScanDir ID'].astype(str)
        else:
            # Try common alternatives
            id_cols = [c for c in merged_df.columns if 'id' in c.lower() or 'subject' in c.lower()]
            if id_cols:
                merged_df['participant_id'] = merged_df[id_cols[0]]
            else:
                print("ERROR: Could not find participant ID column")
                return None
    
    # Standardize label column (DX is diagnosis column: 0=Control, 1=ADHD, others=other diagnoses)
    if 'label' not in merged_df.columns:
        if 'DX' in merged_df.columns:
            # Convert numeric DX: 0=Control, 1=ADHD, rest=drop or convert to control
            merged_df['label'] = merged_df['DX'].apply(lambda x: 1 if x == 1 else 0)
        else:
            label_cols = [c for c in merged_df.columns if 'label' in c.lower() or 'diagnosis' in c.lower()]
            if label_cols:
                merged_df['label'] = merged_df[label_cols[0]]
            else:
                print("ERROR: Could not find diagnosis column")
                return None
    
    # Standardize age column
    if 'age' not in merged_df.columns:
        if 'Age' in merged_df.columns:
            merged_df['age'] = merged_df['Age']
        else:
            age_cols = [c for c in merged_df.columns if 'age' in c.lower()]
            if age_cols:
                merged_df['age'] = merged_df[age_cols[0]]
    
    # Standardize gender column
    if 'gender' not in merged_df.columns:
        if 'Gender' in merged_df.columns:
            merged_df['gender'] = merged_df['Gender']
        else:
            gender_cols = [c for c in merged_df.columns if 'gender' in c.lower() or 'sex' in c.lower()]
            if gender_cols:
                merged_df['gender'] = merged_df[gender_cols[0]]
    
    # Standardize IQ column
    if 'iq' not in merged_df.columns:
        if 'Full4 IQ' in merged_df.columns:
            merged_df['iq'] = merged_df['Full4 IQ']
        elif 'Full2 IQ' in merged_df.columns:
            merged_df['iq'] = merged_df['Full2 IQ']
        else:
            iq_cols = [c for c in merged_df.columns if 'iq' in c.lower() or 'fsiq' in c.lower()]
            if iq_cols:
                merged_df['iq'] = merged_df[iq_cols[0]]
    
    # Select standard columns
    standard_cols = ['participant_id', 'site', 'label', 'age', 'gender']
    if 'iq' in merged_df.columns:
        standard_cols.append('iq')
    
    # Only keep columns that exist
    export_cols = [c for c in standard_cols if c in merged_df.columns]
    participants_df = merged_df[export_cols].copy()
    
    # Remove duplicates by participant_id
    participants_df = participants_df.drop_duplicates(subset=['participant_id'], keep='first')
    
    print(f"✓ Standardized {len(participants_df)} unique subjects")
    print(f"  Columns: {export_cols}\n")
    
    # Display summary
    print("Sample data:")
    print(participants_df.head(10))
    print(f"\nLabel distribution:")
    print(participants_df['label'].value_counts().sort_index())
    print(f"\nSite distribution:")
    print(participants_df['site'].value_counts().sort_index())
    
    return participants_df

def save_participants_tsv(participants_df):
    """
    Save participants.tsv in BIDS format
    """
    print("\n[3] Saving participants.tsv...")
    
    output_path = os.path.join(ADHD_ROOT, "participants.tsv")
    
    # Save with tab separation
    participants_df.to_csv(output_path, sep="\t", index=False)
    
    print(f"✓ Saved {output_path}")
    print(f"  Shape: {participants_df.shape}")
    
    return output_path

def create_bids_anatomy_links():
    """
    Create symbolic links from BIDS structure to original data files
    """
    print("\n[4] Creating BIDS-compliant anatomy links...")
    
    participants_path = os.path.join(ADHD_ROOT, "participants.tsv")
    participants_df = pd.read_csv(participants_path, sep="\t")
    
    created = 0
    failed = 0
    
    for _, row in participants_df.iterrows():
        sub_id = str(row['participant_id'])
        site = row['site']
        
        # Create subject directory structure
        subject_dir = os.path.join(ADHD_ROOT, f"sub-{sub_id}", "anat")
        os.makedirs(subject_dir, exist_ok=True)
        
        # Find the NIfTI file in the original site folder
        site_dir = os.path.join(ADHD_ROOT, site, f"sub-{sub_id}")
        
        if not os.path.exists(site_dir):
            failed += 1
            continue
        
        # Look for normalized_resampled_128 .nii file
        nii_files = [f for f in os.listdir(site_dir) 
                    if "normalized_resampled_128" in f and f.endswith(".nii")]
        
        if not nii_files:
            failed += 1
            continue
        
        src_file = os.path.join(site_dir, nii_files[0])
        dst_file = os.path.join(subject_dir, f"{sub_id}_T1w.nii.gz")
        
        try:
            # Create symlink if it doesn't exist
            if not os.path.exists(dst_file):
                os.symlink(os.path.abspath(src_file), dst_file)
            created += 1
        except Exception as e:
            print(f"  ⚠️  Failed to link sub-{sub_id}: {e}")
            failed += 1
    
    print(f"✓ Created {created} anatomy links")
    if failed > 0:
        print(f"⚠️  Failed to create {failed} links")

def main():
    print("=" * 70)
    print("ADHD200 Data Preparation for BIDS Format")
    print("=" * 70 + "\n")
    
    # Check if ADHD_BIDS exists
    if not os.path.exists(ADHD_ROOT):
        print(f"ERROR: {ADHD_ROOT} not found!")
        print("Please extract the adhd200_128_normalized.zip file first.")
        return
    
    # Merge phenotypic data
    merged_df = merge_phenotypic_data()
    
    # Process subjects
    participants_df = process_subjects(merged_df)
    
    # Save participants.tsv
    save_participants_tsv(participants_df)
    
    # Create BIDS anatomy links
    create_bids_anatomy_links()
    
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)
    print(f"\nYou can now run the training pipeline:")
    print("  python src/sMRI_adhd_pipeline.py")
    print("  python src/train_attention_fusion.py")

if __name__ == "__main__":
    main()
