#!/usr/bin/env python3

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
CELL_ZIP_PATH = './datasets_process/dataset_source/CELL/task1_test_imgs.zip'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# CELL corresponds to categories
TARGET_CATEGORIES = ['Catering-Services', 'Administrative', 'Education']


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


def extract_image_filenames(label_data: Dict) -> Set[str]:
    """Extract all image filenames from label.json"""
    image_filenames = set()
    for key in label_data.keys():
        # Keys are image filenames
        if any(key.lower().endswith(ext.lower()) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_filenames.add(key)
    return image_filenames


def find_image_file(image_filename: str, search_dir: str, *, exact_only: bool = False) -> Optional[str]:
    """Find image by exact name, or stem+ext / contains if not exact_only.
    When exact_only: exact match, then stem+ext only (no contains). Handles .jpg vs .jpeg in CELL."""
    search_path = Path(search_dir)

    # Strategy 1: Exact filename match (always used)
    for img_path in search_path.rglob(image_filename):
        if img_path.is_file():
            return str(img_path)

    # Stem + extension variants: 985.jpg <-> 985.jpeg etc. Safe (same base name only).
    image_stem = Path(image_filename).stem
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for img_path in search_path.rglob(f"{image_stem}{ext}"):
            if img_path.is_file():
                return str(img_path)

    if exact_only:
        return None

    # Strategy 3: Filename contains (skip when exact_only to avoid img_1492->1492, test_28->28)
    image_lower = image_filename.lower()
    for img_path in search_path.rglob('*'):
        if img_path.is_file():
            if image_lower in img_path.name.lower() or img_path.name.lower() in image_lower:
                return str(img_path)

    return None


def copy_images_for_category(category: str, image_filenames: Set[str],
                              source_dir: str, dest_images_dir: str, *,
                              exact_only: bool = False) -> int:
    """Copy images for a single category, return success count.

    When exact_only=True, only copies when CELL has a file with the exact same
    name. Use this for categories that mix CELL with other sources (e.g. Education).
    """
    os.makedirs(dest_images_dir, exist_ok=True)

    success_count = 0

    for image_filename in sorted(image_filenames):
        source_image_path = find_image_file(
            image_filename, source_dir, exact_only=exact_only
        )
        
        if not source_image_path:
            continue
        
        # Target path
        dest_image_path = os.path.join(dest_images_dir, image_filename)
        
        # If target file exists, check if update is needed
        if os.path.exists(dest_image_path):
            if os.path.getsize(dest_image_path) == os.path.getsize(source_image_path):
                success_count += 1
                continue
        
        # Copy file
        try:
            shutil.copy2(source_image_path, dest_image_path)
            success_count += 1
        except Exception:
            pass
    
    return success_count


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
    
    # 2. Process each category
    # All CELL categories are multi-source; use exact_only to avoid overwrites:
    # - Education: EPHOIE (img_*.jpg) + CELL (985–994). Loose match: img_1492 -> 1492.
    # - Catering-Services: CORD (test_*.jpg) + CELL (1010, 1144, ...). Loose: test_28 -> 28.
    # - Administrative: FUNSD (82200067_0069.png) + CELL. Loose: 0069 -> 69, 67 -> 67.
    USE_EXACT_ONLY = True

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

        image_filenames = extract_image_filenames(label_data)

        success = copy_images_for_category(
            category, image_filenames, extract_dir, images_dir,
            exact_only=USE_EXACT_ONLY
        )

        total_success += success
    
    # Output result
    print(f"Success: {total_success}")


if __name__ == '__main__':
    process_cell()
