#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
CORD_REPO = 'naver-clova-ix/cord-v2'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'
# Use mirror site
USE_MIRROR = True  

# CORD corresponds to category
TARGET_CATEGORY = 'Catering-Services'


def download_huggingface_dataset(repo_id: str, target_dir: str) -> Optional[str]:
    """Download specified parquet file from HuggingFace"""
    target_path = os.path.join(DATA_SOURCE_DIR, target_dir)
    
    # Find parquet files starting with data/test
    parquet_pattern = 'data/test'
    
    os.makedirs(DATA_SOURCE_DIR, exist_ok=True)
    
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    
    # Use HfApi to download specified parquet file
    try:
        # If using mirror, set endpoint
        if USE_MIRROR:
            api = HfApi(endpoint='https://hf-mirror.com')
        else:
            api = HfApi()
        
        # List all files first to see actual file structure
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        
        # Find parquet files starting with data/test
        test_parquet_files = [f for f in files if f.startswith(parquet_pattern) and f.endswith('.parquet')]
        
        if not test_parquet_files:
            return None
        
        # Download first found test parquet file
        parquet_filename = test_parquet_files[0]
        parquet_path = os.path.join(target_path, parquet_filename)
        
        if os.path.exists(parquet_path):
            return target_path
        
        # Download parquet file
        api.hf_hub_download(
            repo_id=repo_id,
            filename=parquet_filename,
            repo_type="dataset",
            local_dir=target_path,
            resume_download=True
        )
        
        # Check if download succeeded
        if os.path.exists(parquet_path):
            return target_path
        else:
            return None
            
    except Exception as e:
        return None


def extract_images_from_parquet(parquet_path: str, output_dir: str) -> Dict[str, str]:
    """
    Extract images from parquet file
    Returns: Dictionary of {image filename: saved path}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import pandas as pd
    except ImportError:
        return {}
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        
        image_map = {}
        
        # Extract images
        for idx, row in df.iterrows():
            image_data = row['image']
            
            # Determine filename - use test_index.jpg format
            filename = f"test_{idx}.jpg"
            
            # Save image
            image_path = os.path.join(output_dir, filename)
            
            # Handle different types of image data
            image_bytes = None
            
            if isinstance(image_data, bytes):
                # Direct byte data
                image_bytes = image_data
            elif isinstance(image_data, dict):
                # If dict, try to extract image data
                if 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                elif 'data' in image_data:
                    image_bytes = image_data['data']
                elif 'image' in image_data:
                    image_bytes = image_data['image']
                else:
                    # Try to get first value
                    if len(image_data) > 0:
                        first_value = list(image_data.values())[0]
                        if isinstance(first_value, bytes):
                            image_bytes = first_value
                        else:
                            continue
                    else:
                        continue
            elif isinstance(image_data, str):
                # May be base64 encoded
                if len(image_data) > 1000:
                    try:
                        import base64
                        # Remove data:image/xxx;base64, prefix
                        if ',' in image_data:
                            image_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                    except Exception:
                        continue
                else:
                    continue
            else:
                continue
            
            # Save image byte data
            if image_bytes:
                try:
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_map[filename] = image_path
                except Exception:
                    continue
        
        return image_map
        
    except Exception as e:
        return {}


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


def find_image_file(image_filename: str, image_map: Dict[str, str]) -> Optional[str]:
    """Find image file in image map"""
    # Strategy 1: Exact filename match
    if image_filename in image_map:
        return image_map[image_filename]
    
    # Strategy 2: Filename (without extension) match
    image_stem = Path(image_filename).stem
    for filename, path in image_map.items():
        if Path(filename).stem == image_stem:
            return path
    
    # Strategy 3: Filename contains relationship (case-insensitive)
    image_lower = image_filename.lower()
    for filename, path in image_map.items():
        if image_lower in filename.lower() or filename.lower() in image_lower:
            return path
    
    return None


def copy_images_for_category(category: str, image_filenames: Set[str], 
                              image_map: Dict[str, str], dest_images_dir: str) -> tuple:
    """Copy images for a single category"""
    os.makedirs(dest_images_dir, exist_ok=True)
    
    success_count = 0
    fail_count = 0
    failed_files = []
    
    for image_filename in sorted(image_filenames):
        # Find source image file
        source_image_path = find_image_file(image_filename, image_map)
        
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


def process_cord():
    """Main processing workflow"""
    # 1. Download parquet file
    target_dir = 'CORD'
    downloaded_dir = download_huggingface_dataset(CORD_REPO, target_dir)
    
    if not downloaded_dir:
        return
    
    # Find downloaded parquet file
    parquet_files = list(Path(downloaded_dir).rglob('data/test*.parquet'))
    if not parquet_files:
        return
    
    parquet_path = str(parquet_files[0])
    
    # 2. Extract images
    images_dir = os.path.join(downloaded_dir, 'images')
    image_map = extract_images_from_parquet(parquet_path, images_dir)
    
    if not image_map:
        return
    
    # 3. Process specified category
    category_dir = os.path.join(DATASETS_ROOT, TARGET_CATEGORY)
    label_path = os.path.join(category_dir, 'label.json')
    dest_images_dir = os.path.join(category_dir, 'images')
    
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
        TARGET_CATEGORY, image_filenames, image_map, dest_images_dir
    )
    
    # Output result
    print(f"Success: {success}")


if __name__ == '__main__':
    process_cord()
