#!/usr/bin/env python3

import os
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
SROIE_ZIP_PATH = './datasets_process/dataset_source/SROIE/SROIE_test_images_task_3.zip'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# SROIE corresponds to category
TARGET_CATEGORY = 'Retail'


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


def find_image_file(image_filename: str, search_dir: str) -> Optional[str]:
    """Find image file in search directory"""
    search_path = Path(search_dir)
    
    # Strategy 1: Exact filename match
    for img_path in search_path.rglob(image_filename):
        if img_path.is_file():
            return str(img_path)
    
    # Strategy 2: Filename (without extension) match
    image_stem = Path(image_filename).stem
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for img_path in search_path.rglob(f"{image_stem}{ext}"):
            if img_path.is_file():
                return str(img_path)
    
    # Strategy 3: Filename contains relationship (case-insensitive)
    image_lower = image_filename.lower()
    for img_path in search_path.rglob('*'):
        if img_path.is_file():
            if image_lower in img_path.name.lower() or img_path.name.lower() in image_lower:
                return str(img_path)
    
    return None


def copy_images_for_category(category: str, image_filenames: Set[str], 
                              source_dir: str, dest_images_dir: str) -> tuple:
    """Copy images for a single category"""
    os.makedirs(dest_images_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    for image_filename in sorted(image_filenames):
        # Find source image file
        source_image_path = find_image_file(image_filename, source_dir)
        
        if not source_image_path:
            fail_count += 1
            failed_files.append(image_filename)
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
            fail_count += 1
            failed_files.append(image_filename)
    
    return success_count, fail_count, failed_files


def process_sroie():
    """Main processing workflow"""
    # 1. Extract zip file
    extract_dir = os.path.join(DATA_SOURCE_DIR, 'SROIE', 'extracted')
    
    # Check if already extracted (check if there are image files)
    has_images = (os.path.exists(extract_dir) and 
                  (any(Path(extract_dir).rglob('*.jpg')) or any(Path(extract_dir).rglob('*.png'))))
    
    if not has_images:
        if not extract_archive(SROIE_ZIP_PATH, extract_dir):
            return
    
    # 2. Load label.json
    category_dir = os.path.join(DATASETS_ROOT, TARGET_CATEGORY)
    label_path = os.path.join(category_dir, 'label.json')
    images_dir = os.path.join(category_dir, 'images')
    
    if not os.path.exists(label_path):
        return
    
    label_data = load_label_json(label_path)
    if not label_data:
        return
    
    # Extract image filenames
    image_filenames = extract_image_filenames(label_data)
    
    # Copy images
    success, fail, failed_files = copy_images_for_category(
        TARGET_CATEGORY, image_filenames, extract_dir, images_dir
    )
    
    # Output result
    print(f"Success: {success}")


if __name__ == '__main__':
    process_sroie()
