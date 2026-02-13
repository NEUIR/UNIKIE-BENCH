#!/usr/bin/env python3

import os
import json
import shutil
import zipfile
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
FUNSD_DATASET_URL = 'https://guillaumejaume.github.io/FUNSD/dataset.zip'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# FUNSD corresponds to category
TARGET_CATEGORY = 'Administrative'


def download_file(url: str, target_path: str) -> bool:
    """Download file"""
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        response = requests.get(url, stream=True, verify=False, timeout=300)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(target_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
        
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            return True
    except ImportError:
        pass
    except Exception as e:
        return False
    
    return False


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
    except Exception as e:
        return False


def find_and_extract_archives(directory: str) -> bool:
    """Find and extract all archive files in directory"""
    directory_path = Path(directory)
    archives = []
    
    # Find all archive files
    for ext in ['.zip', '.tar.gz', '.tgz', '.tar']:
        archives.extend(directory_path.rglob(f'*{ext}'))
    
    if not archives:
        return True
    
    success = True
    for archive in archives:
        if not extract_archive(str(archive), directory):
            success = False
    
    return success


def load_label_json(label_path: str) -> Dict:
    """Load label.json file"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
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
        except Exception as e:
            fail_count += 1
            failed_files.append(image_filename)
    
    return success_count, fail_count, failed_files


def process_funsd():
    """Main processing workflow"""
    # 1. Download file
    target_dir = os.path.join(DATA_SOURCE_DIR, 'FUNSD')
    os.makedirs(target_dir, exist_ok=True)
    
    # Try different filenames
    archive_names = ['dataset.zip', 'FUNSD.zip', 'funsd.zip', 'FUNSD.tar.gz', 'funsd.tar.gz']
    archive_path = None
    
    for archive_name in archive_names:
        potential_path = os.path.join(target_dir, archive_name)
        if os.path.exists(potential_path):
            archive_path = potential_path
            break
    
    if not archive_path:
        # Download file
        archive_path = os.path.join(target_dir, 'dataset.zip')
        if not download_file(FUNSD_DATASET_URL, archive_path):
            return
    
    # 2. Extract file
    extract_dir = os.path.join(target_dir, 'extracted')
    
    if not (os.path.exists(extract_dir) and (any(Path(extract_dir).rglob('*.jpg')) or any(Path(extract_dir).rglob('*.png')))):
        if not extract_archive(archive_path, extract_dir):
            return
        
        # If there are still archive files after extraction, continue extracting
        find_and_extract_archives(extract_dir)
    
    # 3. Process Administrative category
    
    category_dir = os.path.join(DATASETS_ROOT, TARGET_CATEGORY)
    label_path = os.path.join(category_dir, 'label.json')
    images_dir = os.path.join(category_dir, 'images')
    
    if not os.path.exists(label_path):
        return
    
    # Load label.json
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
    process_funsd()
