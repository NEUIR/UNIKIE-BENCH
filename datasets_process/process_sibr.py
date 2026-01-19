#!/usr/bin/env python3
import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
SIBR_GIT_URL = 'https://www.modelscope.cn/datasets/iic/SIBR.git'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# SIBR corresponds to three categories
SIBR_CATEGORIES = ['Accommodation', 'Medical-Services', 'Commercial']


def run_command(cmd: List[str], cwd: Optional[str] = None) -> bool:
    """Run command and return success status"""
    try:
        print(f"Executing command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"Error executing command: {e}")
        return False


def download_sibr() -> str:
    """Download SIBR dataset"""
    sibr_dir = os.path.join(DATA_SOURCE_DIR, 'SIBR')
    
    if os.path.exists(sibr_dir):
        print(f"SIBR directory already exists: {sibr_dir}")
        print("Skipping download. Delete the directory to re-download.")
        return sibr_dir
    
    print(f"Starting SIBR dataset download...")
    print(f"Git URL: {SIBR_GIT_URL}")
    print(f"Target directory: {sibr_dir}")
    
    os.makedirs(DATA_SOURCE_DIR, exist_ok=True)
    
    cmd = ['git', 'clone', SIBR_GIT_URL, sibr_dir]
    if run_command(cmd):
        print(f"✓ SIBR download completed")
        return sibr_dir
    else:
        print(f"✗ SIBR download failed")
        return None


def find_images_zip(sibr_dir: str) -> str:
    """Find images.zip file"""
    print(f"\nSearching for images.zip file...")
    
    sibr_path = Path(sibr_dir)
    zip_files = list(sibr_path.rglob('images.zip'))
    
    if not zip_files:
        print(f"images.zip file not found")
        print(f"Search path: {sibr_dir}")
        return None
    
    if len(zip_files) > 1:
        print(f"Found multiple images.zip files:")
        for zf in zip_files:
            print(f"  - {zf}")
        print(f"Using the first one: {zip_files[0]}")
    
    zip_path = str(zip_files[0])
    print(f"✓ Found images.zip: {zip_path}")
    return zip_path


def extract_images_zip(zip_path: str, extract_to: str) -> bool:
    """Extract images.zip"""
    print(f"\nExtracting images.zip...")
    print(f"Source file: {zip_path}")
    print(f"Extract to: {extract_to}")
    
    if not os.path.exists(zip_path):
        print(f"Error: File does not exist: {zip_path}")
        return False
    
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get file list
            file_list = zip_ref.namelist()
            print(f"Archive contains {len(file_list)} files")
            
            # Extract all files
            zip_ref.extractall(extract_to)
            print(f"✓ Extraction completed")
            return True
    except zipfile.BadZipFile:
        print(f"Error: Not a valid zip file")
        return False
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def load_label_json(label_path: str) -> Dict:
    """Load label.json file"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error: Cannot read {label_path}: {e}")
        return {}


def extract_image_filenames(label_data: Dict) -> Set[str]:
    """Extract all image filenames from label.json"""
    image_filenames = set()
    for key in label_data.keys():
        # Keys are image filenames
        if any(key.lower().endswith(ext.lower()) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_filenames.add(key)
    return image_filenames


def find_image_file(image_filename: str, search_dir: str) -> str:
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


def process_sibr():
    """Main processing workflow"""
    print("="*60)
    print("SIBR Dataset Processing Script")
    print("="*60)
    
    # 1. Download SIBR
    print("\nStep 1: Download SIBR dataset")
    sibr_dir = download_sibr()
    if not sibr_dir:
        print("Download failed, exiting")
        return
    
    # 2. Find images.zip
    print("\nStep 2: Find images.zip")
    zip_path = find_images_zip(sibr_dir)
    if not zip_path:
        print("images.zip not found, exiting")
        return
    
    # 3. Extract images.zip
    print("\nStep 3: Extract images.zip")
    extract_dir = os.path.join(DATA_SOURCE_DIR, 'SIBR_images')
    if not extract_images_zip(zip_path, extract_dir):
        print("Extraction failed, exiting")
        return
    
    # 4. Process each category
    print("\nStep 4: Copy images to respective categories based on label.json")
    print("="*60)
    
    total_success = 0
    total_fail = 0
    all_failed_files = {}
    
    for category in SIBR_CATEGORIES:
        category_dir = os.path.join(DATASETS_ROOT, category)
        label_path = os.path.join(category_dir, 'label.json')
        images_dir = os.path.join(category_dir, 'images')
        
        if not os.path.exists(label_path):
            print(f"\nWarning: {category} label.json not found: {label_path}")
            continue
        
        # Load label.json
        label_data = load_label_json(label_path)
        if not label_data:
            print(f"\nWarning: {category} label.json is empty")
            continue
        
        # Extract image filenames
        image_filenames = extract_image_filenames(label_data)
        print(f"\n{category}: Found {len(image_filenames)} image filenames")
        
        # Copy images
        success, fail, failed_files = copy_images_for_category(
            category, image_filenames, extract_dir, images_dir
        )
        
        total_success += success
        total_fail += fail
        if failed_files:
            all_failed_files[category] = failed_files
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total: Success {total_success}")
    
    if all_failed_files:
        print(f"\nFailed files:")
        for category, files in all_failed_files.items():
            print(f"\n{category} ({len(files)} files):")
            for f in files[:10]:  # Show only first 10
                print(f"  - {f}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
    
    print(f"\n✓ Processing completed!")
    print(f"Images copied to respective category images folders")
    print(f"Source files retained at: {extract_dir}")


if __name__ == '__main__':
    process_sibr()
