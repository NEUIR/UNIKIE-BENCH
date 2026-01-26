#!/usr/bin/env python3

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Optional

from hash_utils import copy_sources_to_dest_by_hash

# Configuration
CELL_ZIP_PATH = './datasets_process/dataset_source/CELL/task1_test_imgs.zip'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# CELL corresponds to categories
TARGET_CATEGORIES = ['Catering-Services', 'Administrative', 'Education']

EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
EXCLUDE_DIRS = {'__MACOSX'}


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Extract archive file"""
    if not os.path.exists(archive_path):
        return False
    
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
            import tarfile
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)
            return True
        elif archive_path.endswith('.tar'):
            import tarfile
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
            return True
        else:
            return False
    except Exception:
        return False


def load_label_json(label_path: str) -> Dict:
    """Load label.json file"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _collect_cell_sources(extract_dir: str) -> List[Path]:
    """Collect all image paths under CELL extract, excluding __MACOSX etc."""
    root = Path(extract_dir)
    out = []
    for p in root.rglob('*'):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        if any(d in p.parts for d in EXCLUDE_DIRS):
            continue
        out.append(p)
    return out


def _image_label_keys(label_data: Dict) -> Set[str]:
    """Label keys that are image-like (hash.ext). Excludes e.g. Commercial hex dirs."""
    return {
        k for k in label_data.keys()
        if any(k.lower().endswith(e) for e in EXTS)
    }


def process_cell():
    """Main processing workflow"""
    # 1. Extract zip file
    extract_dir = os.path.join(DATA_SOURCE_DIR, 'CELL', 'extracted')
    
    # Check if already extracted (check if there are image files)
    has_images = (os.path.exists(extract_dir) and 
                  (any(Path(extract_dir).rglob('*.jpg')) or any(Path(extract_dir).rglob('*.png'))))
    
    if not has_images:
        if not extract_archive(CELL_ZIP_PATH, extract_dir):
            return
    
    # 2. Collect all CELL source images (hash-based: match by content hash to label keys)
    sources = _collect_cell_sources(extract_dir)
    if not sources:
        return

    total_success = 0
    for category in TARGET_CATEGORIES:
        category_dir = os.path.join(DATASETS_ROOT, category)
        label_path = os.path.join(category_dir, 'label.json')
        images_dir = os.path.join(category_dir, 'images')

        if not os.path.exists(label_path):
            continue

        label_data = load_label_json(label_path)
        if not label_data:
            continue

        label_keys = _image_label_keys(label_data)
        success = copy_sources_to_dest_by_hash(
            sources,
            Path(images_dir),
            label_keys,
        )
        total_success += success
    
    # Output result
    print(f"Success: {total_success}")


if __name__ == '__main__':
    process_cell()
