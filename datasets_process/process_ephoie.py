#!/usr/bin/env python3

import os
import sys
import json
import shutil
import csv
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'
USE_MIRROR = True  

# EPHOIE corresponds to category
TARGET_CATEGORY = 'Education'




def download_huggingface_tsv(repo_id: str, tsv_file: str, target_dir: str) -> Optional[str]:
    """Download TSV file from HuggingFace"""
    target_path = os.path.join(DATA_SOURCE_DIR, target_dir)
    os.makedirs(target_path, exist_ok=True)
    
    # TSV filename (without path)
    tsv_filename = os.path.basename(tsv_file)
    tsv_path = os.path.join(target_path, tsv_filename)
    
    # Check if file already exists (including symlinks)
    if os.path.exists(tsv_path) or os.path.islink(tsv_path):
        # If symlink, check if actual file exists
        if os.path.islink(tsv_path):
            actual_path = os.path.realpath(tsv_path)
            if os.path.exists(actual_path):
                return tsv_path
        else:
            return tsv_path
    
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return None
    
    try:
        # If using mirror, set endpoint
        if USE_MIRROR:
            api = HfApi(endpoint='https://hf-mirror.com')
        else:
            api = HfApi()
        
        # Download TSV file (don't use local_dir to avoid creating folder structure)
        temp_path = api.hf_hub_download(
            repo_id=repo_id,
            filename=tsv_file,
            repo_type="dataset",
            force_download=False
        )
        
        # Copy file to target location (keep only filename, don't preserve path structure)
        # If symlink, need to read actual file content
        if os.path.exists(temp_path):
            if os.path.islink(temp_path):
                # If symlink, read actual file content
                with open(temp_path, 'rb') as src:
                    with open(tsv_path, 'wb') as dst:
                        dst.write(src.read())
            else:
                # Directly copy file
                shutil.copy2(temp_path, tsv_path)
            
            if os.path.exists(tsv_path):
                return tsv_path
            else:
                return None
        else:
            return None
            
    except Exception as e:
        return None


def extract_images_from_tsv(tsv_path: str, output_dir: str) -> Dict[str, str]:
    """
    Extract images from TSV file
    Returns: Dictionary of {image filename: image file path}
    """
    os.makedirs(output_dir, exist_ok=True)
    
    image_map = {}
    
    try:
        # Increase CSV field size limit (base64 image data can be very large)
        csv.field_size_limit(sys.maxsize)
        
        # If symlink, use actual path
        if os.path.islink(tsv_path):
            actual_path = os.path.realpath(tsv_path)
        else:
            actual_path = tsv_path
        
        # Try to open file (prefer symlink path, Python will follow automatically)
        try:
            f = open(tsv_path, 'r', encoding='utf-8')
        except (FileNotFoundError, OSError):
            # If symlink path fails, try actual path
            if actual_path != tsv_path:
                try:
                    f = open(actual_path, 'r', encoding='utf-8')
                except (FileNotFoundError, OSError):
                    return image_map
            else:
                return image_map
        
        with f:
            # Try to detect delimiter
            first_line = f.readline()
            f.seek(0)
            
            # TSV files usually use tab delimiter
            delimiter = '\t' if '\t' in first_line else ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            # Get column names
            fieldnames = reader.fieldnames
            if not fieldnames:
                return image_map
            
            # Find image column (usually 'image' column)
            image_key = None
            for key in ['image', 'image_data', 'image_base64']:
                if key in fieldnames:
                    image_key = key
                    break
            
            # If not found, use first column
            if not image_key:
                image_key = fieldnames[0]
            
            # Find image name column (for filename)
            image_name_key = None
            for key in ['image_name', 'filename', 'file_name']:
                if key in fieldnames:
                    image_name_key = key
                    break
            
            for idx, row in enumerate(reader):
                if image_key not in row:
                    continue
                
                image_value = row[image_key].strip()
                if not image_value:
                    continue
                
                # Use image_name column as filename, if not available use index
                if image_name_key and image_name_key in row and row[image_name_key]:
                    filename = row[image_name_key].strip()
                else:
                    filename = f"img_{idx}.jpg"
                
                image_path = os.path.join(output_dir, filename)
                
                # If image already exists, skip
                if os.path.exists(image_path):
                    image_map[filename] = image_path
                    continue
                
                # Process image data (base64 encoded)
                image_bytes = None
                
                try:
                    import base64
                    # base64 data can be very long, decode directly
                    image_bytes = base64.b64decode(image_value)
                except Exception:
                    # If decode fails, try removing possible data:image prefix
                    try:
                        if ',' in image_value:
                            image_value = image_value.split(',', 1)[1]
                        image_bytes = base64.b64decode(image_value)
                    except Exception:
                        continue
                
                # Save image
                if image_bytes:
                    try:
                        with open(image_path, 'wb') as img_file:
                            img_file.write(image_bytes)
                        image_map[filename] = image_path
                    except Exception:
                        continue
                        
    except Exception as e:
        pass
    
    return image_map


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
                              image_map: Dict[str, str], dest_images_dir: str) -> int:
    """Copy images for a single category"""
    os.makedirs(dest_images_dir, exist_ok=True)
    
    success_count = 0
    
    for image_filename in sorted(image_filenames):
        # Target path
        dest_image_path = os.path.join(dest_images_dir, image_filename)
        
        # If target file exists, skip
        if os.path.exists(dest_image_path):
            success_count += 1
            continue
        
        # Find image from image_map (exact filename match)
        if image_filename in image_map:
            source_image_path = image_map[image_filename]
            
            # If source_image_path is a file path, copy directly
            if os.path.exists(source_image_path) and os.path.isfile(source_image_path):
                try:
                    shutil.copy2(source_image_path, dest_image_path)
                    success_count += 1
                except Exception:
                    pass
    
    return success_count


def process_ephoie():
    """Main processing workflow"""
    # Configuration
    repo_id = 'wulipc/CC-OCR'
    tsv_file = 'kie/constrained_category/EPHOIE_SCUT_311.tsv'
    
    # 1. Download TSV file
    target_dir = 'EPHOIE'
    tsv_path = download_huggingface_tsv(repo_id, tsv_file, target_dir)
    
    # Check if file exists (including symlinks)
    if not tsv_path:
        return
    
    # Ensure tsv_path is absolute path
    if not os.path.isabs(tsv_path):
        tsv_path = os.path.abspath(tsv_path)
    
    # 2. Extract image information from TSV file
    # Note: Even if file is symlink pointing outside sandbox, Python's open() will try to follow it
    dataset_path = os.path.dirname(tsv_path)
    extract_dir = os.path.join(dataset_path, 'extracted_images')
    image_map = extract_images_from_tsv(tsv_path, extract_dir)
    
    # If no images extracted, return directly
    if not image_map:
        return
    
    # 3. Load label.json
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
    
    # 4. Copy images
    success_count = copy_images_for_category(
        TARGET_CATEGORY, image_filenames, image_map, images_dir
    )
    
    # Output result
    print(f"Success: {success_count}")


if __name__ == '__main__':
    process_ephoie()
