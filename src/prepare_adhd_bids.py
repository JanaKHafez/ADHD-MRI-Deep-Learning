#!/usr/bin/env python3
"""Prepare a BIDS-like `ADHD_BIDS` folder from the provided ADHD-200 archive.

Usage:
  python src/prepare_adhd_bids.py /path/to/adhd200_128_normalized.zip

This script will:
- Read phenotypic CSVs from the archive to determine diagnosis (DX) per subject.
- Extract each subject volume and place it at `ADHD_BIDS/sub-<id>/anat/<id>_T1w.nii.gz`.
- Create `ADHD_BIDS/participants.tsv` with columns: `participant_id\tsite\tlabel` where
  `label` = 1 if DX != 0 else 0.

This is a convenience conversion so `src/sMRI_adhd_pipeline.py` can run unchanged.
"""
import sys
import os
import re
import csv
import gzip
from zipfile import ZipFile


def build_participant_map(zf):
    """Parse phenotypic CSVs in the archive to build id -> (site, dx) mapping."""
    mapping = {}
    # Look for files ending with '_phenotypic.csv' under adhd200-preprocessed/
    for name in zf.namelist():
        if name.endswith('_phenotypic.csv') and name.startswith('adhd200-preprocessed/'):
            # site can be inferred from filename prefix (e.g., NeuroIMAGE_phenotypic.csv)
            site = os.path.basename(name).replace('_phenotypic.csv', '')
            with zf.open(name) as fh:
                # decode lines as text
                text_lines = [l.decode('utf-8', errors='replace') for l in fh.read().splitlines() if l.strip()]
                if not text_lines:
                    continue
                reader = csv.reader(text_lines)
                header = next(reader, None)
                for row in reader:
                    if not row:
                        continue
                    # First column is ScanDir ID (subject id), DX is at column index 5 per observation
                    try:
                        subj = row[0].strip()
                    except Exception:
                        continue
                    dx = None
                    if len(row) > 5:
                        try:
                            dx = int(row[5])
                        except Exception:
                            # try to clean
                            try:
                                dx = int(row[5].strip())
                            except Exception:
                                dx = None
                    mapping[subj] = {'site': site, 'dx': dx}
    return mapping


def extract_and_rename(zip_path, out_root='ADHD_BIDS'):
    if not os.path.exists(zip_path):
        raise FileNotFoundError(zip_path)

    os.makedirs(out_root, exist_ok=True)

    with ZipFile(zip_path, 'r') as zf:
        mapping = build_participant_map(zf)

        # regex to match subject T1 files used in this archive
        pattern = re.compile(r'.*/sub-(?P<id>\d+)/normalized_resampled_128_sub-(?P=id)_T1_biascorr_brain\.nii$')

        processed = []
        for name in zf.namelist():
            m = pattern.match(name)
            if not m:
                continue
            subj = m.group('id')
            site = None
            if subj in mapping:
                site = mapping[subj]['site']
            # create target dir
            target_dir = os.path.join(out_root, f'sub-{subj}', 'anat')
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f'{subj}_T1w.nii.gz')

            # extract file bytes and gzip to target
            with zf.open(name) as src, gzip.open(target_path, 'wb') as dst:
                data = src.read()
                dst.write(data)

            processed.append({'participant_id': subj, 'site': site if site else 'Unknown'})

        # write participants.tsv based on processed list and mapping dx info
        pth = os.path.join(out_root, 'participants.tsv')
        with open(pth, 'w', newline='') as out_f:
            writer = csv.writer(out_f, delimiter='\t')
            writer.writerow(['participant_id', 'site', 'label'])
            for entry in processed:
                pid = entry['participant_id']
                site = entry['site']
                dx = None
                if pid in mapping:
                    dx = mapping[pid].get('dx')
                # label: 1 if dx is not 0 and not None, else 0
                label = 1 if (dx is not None and dx != 0) else 0
                writer.writerow([pid, site, label])

    print(f"Extraction complete — created '{out_root}' with {len(processed)} subjects and participants.tsv")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python src/prepare_adhd_bids.py /path/to/adhd200_128_normalized.zip')
        sys.exit(1)
    zip_path = sys.argv[1]
    extract_and_rename(zip_path)
