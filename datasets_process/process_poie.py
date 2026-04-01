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
POIE_GDRIVE_FILE_ID = '1eEMNiVeLlD-b08XW_GfAGfPmmII-GDYs'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# POIE corresponds to category
TARGET_CATEGORY = 'Nutrition-Label'


def download_gdrive_file(file_id: str, target_path: str) -> bool:
    """Download Google Drive file"""
    print(f"Downloading Google Drive file: {file_id}")
    
    # Method 1: Use gdown Python library (recommended for Google Drive)
    try:
        import gdown
        print("Using gdown library (recommended method)...")
        # Use the full Google Drive URL format that gdown handles better
        url = f'https://drive.google.com/uc?id={file_id}'
        # gdown handles large files and virus scan warnings automatically
        gdown.download(url, target_path, quiet=False, fuzzy=True)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"✓ Download completed: {target_path}")
            print(f"File size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
            return True
    except ImportError:
        print("gdown Python package not installed, trying other methods...")
    except Exception as e:
        print(f"gdown download failed: {e}")
        print("Trying other methods...")
    
    # Method 2: Use command line gdown
    if shutil.which('gdown'):
        print("Using command line gdown...")
        url = f'https://drive.google.com/uc?id={file_id}'
        cmd = ['gdown', url, '-O', target_path, '--fuzzy']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode == 0 and os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"✓ Download completed: {target_path}")
            print(f"File size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
            return True
        else:
            print(f"gdown command line failed:")
            if result.stdout:
                print(f"  Output: {result.stdout}")
            if result.stderr:
                print(f"  Error: {result.stderr}")
    
    # Method 3: Use requests with export=download (for large files)
    try:
        import requests
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        print("Using requests to download (handling large files)...")
        # For large files, use export=download
        url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # First request to get the confirmation token for large files
        session = requests.Session()
        response = session.get(url, stream=True, verify=False, timeout=30)
        
        # Check if we got a virus scan warning (large files)
        if 'virus scan warning' in response.text.lower() or 'download anyway' in response.text.lower():
            print("Large file detected, confirmation required...")
            # Extract confirmation token
            import re
            confirm_token_match = re.search(r'confirm=([^&]+)', response.text)
            if confirm_token_match:
                confirm_token = confirm_token_match.group(1)
                url = f'https://drive.google.com/uc?export=download&id={file_id}&confirm={confirm_token}'
                response = session.get(url, stream=True, verify=False, timeout=30)
        
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            print(f"File size: {total_size / 1024 / 1024:.2f} MB")
        
        with open(target_path, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192 * 8):  # 64KB chunks
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        if downloaded % (10 * 1024 * 1024) == 0:  # Print every 10MB
                            print(f"  Downloaded: {downloaded / 1024 / 1024:.2f} MB ({percent:.1f}%)")
        
        if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
            print(f"✓ Download completed: {target_path}")
            print(f"File size: {os.path.getsize(target_path) / 1024 / 1024:.2f} MB")
            return True
    except ImportError:
        print("requests not installed")
    except Exception as e:
        print(f"requests download failed: {e}")
    
    print("\nError: All download methods failed")
    print("Suggestions:")
    print("1. Install gdown (recommended): pip install gdown")
    print("2. Or install requests: pip install requests")
    print(f"3. Or manually download: https://drive.google.com/file/d/{file_id}/view")
    print(f"   Save to: {target_path}")
    return False


def extract_archive(archive_path: str, extract_to: str) -> bool:
    """Extract archive file"""
    if not os.path.exists(archive_path):
        print(f"File does not exist: {archive_path}")
        return False
    
    print(f"Extracting: {archive_path} -> {extract_to}")
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
            print(f"Unsupported archive format: {archive_path}")
            return False
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def find_and_extract_archives(directory: str) -> bool:
    """Find and extract all archive files in directory"""
    directory_path = Path(directory)
    archives = []
    
    # Find all archive files
    for ext in ['.zip', '.tar.gz', '.tgz', '.tar']:
        archives.extend(directory_path.rglob(f'*{ext}'))
    
    if not archives:
        # No archive files is normal, no need to output info
        return True
    
    success = True
    for archive in archives:
        print(f"Found archive file: {archive}")
        if not extract_archive(str(archive), directory):
            success = False
    
    return success


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


def process_poie():
    """Main processing workflow"""
    print("="*60)
    print("POIE Dataset Processing Script")
    print("="*60)
    
    # 1. Download file
    print("\nStep 1: Downloading POIE dataset")
    target_dir = os.path.join(DATA_SOURCE_DIR, 'POIE')
    os.makedirs(target_dir, exist_ok=True)
    
    # Try different filenames
    archive_names = ['POIE.zip', 'poie.zip', 'POIE.tar.gz', 'poie.tar.gz']
    archive_path = None
    
    for archive_name in archive_names:
        potential_path = os.path.join(target_dir, archive_name)
        if os.path.exists(potential_path):
            print(f"Found existing file: {potential_path}")
            archive_path = potential_path
            break
    
    if not archive_path:
        # Download file
        archive_path = os.path.join(target_dir, 'POIE.zip')
        if not download_gdrive_file(POIE_GDRIVE_FILE_ID, archive_path):
            print("Download failed, exiting")
            return
    
    # 2. Extract file
    print("\nStep 2: Extracting file")
    extract_dir = os.path.join(target_dir, 'extracted')
    
    if os.path.exists(extract_dir) and (any(Path(extract_dir).rglob('*.jpg')) or any(Path(extract_dir).rglob('*.png'))):
        print(f"Extraction directory already exists and contains images: {extract_dir}")
    else:
        if not extract_archive(archive_path, extract_dir):
            print("Extraction failed, exiting")
            return
        
        # If there are still archive files after extraction, continue extracting
        find_and_extract_archives(extract_dir)
    
    # 3. Process Nutrition-Label category
    print("\nStep 3: Copying images to Nutrition-Label category based on label.json")
    print("="*60)
    
    category_dir = os.path.join(DATASETS_ROOT, TARGET_CATEGORY)
    label_path = os.path.join(category_dir, 'label.json')
    images_dir = os.path.join(category_dir, 'images')
    
    if not os.path.exists(label_path):
        print(f"Error: label.json not found for {TARGET_CATEGORY}: {label_path}")
        return
    
    # Load label.json
    label_data = load_label_json(label_path)
    if not label_data:
        print(f"Error: label.json is empty for {TARGET_CATEGORY}")
        return
    
    # Extract image filenames
    image_filenames = extract_image_filenames(label_data)
    print(f"\n{TARGET_CATEGORY}: Found {len(image_filenames)} image filenames")
    
    # Copy images
    success, fail, failed_files = copy_images_for_category(
        TARGET_CATEGORY, image_filenames, extract_dir, images_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total: {success} successful")
    
    if failed_files:
        print(f"\nFailed files:")
        for f in failed_files[:20]:  # Show first 20
            print(f"  - {f}")
        if len(failed_files) > 20:
            print(f"  ... and {len(failed_files) - 20} more files")
    
    print(f"\n✓ Processing completed!")
    print(f"Images copied to {TARGET_CATEGORY} category images folder")
    print(f"Source files retained at: {extract_dir}")


if __name__ == '__main__':
    process_poie()
