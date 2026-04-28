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
NANONETS_KIE_REPO = 'nanonets/key_information_extraction'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'
# Use mirror site
USE_MIRROR = True  # Use hf-mirror.com mirror

# Nanonets-KIE corresponds to category
TARGET_CATEGORY = 'Tax-Compliant'


def download_huggingface_dataset(repo_id: str, target_dir: str) -> Optional[str]:
    """Download dataset from HuggingFace"""
    target_path = os.path.join(DATA_SOURCE_DIR, target_dir)
    
    if os.path.exists(target_path):
        print(f"Directory already exists: {target_path}")
        print("Skipping download. Delete the directory to re-download.")
        return target_path
    
    print(f"Starting HuggingFace dataset download: {repo_id}")
    print(f"Target directory: {target_path}")
    
    os.makedirs(DATA_SOURCE_DIR, exist_ok=True)
    
    try:
        from huggingface_hub import HfApi
    except ImportError as e:
        import sys
        print("Error: huggingface_hub not installed")
        print(f"Python path: {sys.executable}")
        print("Please install: pip install huggingface_hub")
        print(f"Detailed error: {e}")
        return None
    
    # Use HfApi to list files and download one by one
    print("\nDownloading using HfApi...")
    try:
        # If using mirror, set endpoint
        if USE_MIRROR:
            api = HfApi(endpoint='https://hf-mirror.com')
            print("Using mirror site: hf-mirror.com")
        else:
            api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"Found {len(files)} files")
        
        if not files:
            print("No files found")
            return None
        
        # Only download parquet files
        parquet_files = [f for f in files if f.endswith('.parquet')]
        if not parquet_files:
            print("No parquet files found, downloading all files...")
            parquet_files = files
        
        print(f"Will download {len(parquet_files)} files")
        
        for file_path in parquet_files:
            print(f"Downloading: {file_path}")
            try:
                downloaded_path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=target_path,
                    local_dir_use_symlinks=False,
                    resume_download=True
                )
                print(f"  ✓ {file_path}")
            except Exception as e:
                print(f"  ✗ {file_path}: {e}")
        
        # Check if download succeeded
        if os.path.exists(target_path) and any(Path(target_path).rglob('*.parquet')):
            print(f"✓ Download completed")
            return target_path
        else:
            print("No parquet files found in downloaded files")
            return None
            
    except Exception as e:
        print(f"Download failed: {e}")
        print("\nSuggestions:")
        print("1. Check network connection")
        print("2. Try manually accessing: https://hf-mirror.com/datasets/" + repo_id)
        print("3. Manually download and place at: " + target_path)
        return None


def find_parquet_files(dataset_dir: str) -> List[str]:
    """Find all parquet files"""
    print(f"\nSearching for parquet files...")
    
    dataset_path = Path(dataset_dir)
    parquet_files = list(dataset_path.rglob('*.parquet'))
    
    if not parquet_files:
        print(f"No parquet files found")
        return []
    
    print(f"Found {len(parquet_files)} parquet files:")
    for pf in parquet_files:
        print(f"  - {pf}")
    
    return [str(pf) for pf in parquet_files]


def extract_images_from_parquet(parquet_path: str, output_dir: str) -> Dict[str, str]:
    """
    Extract images from parquet file
    Returns: Dictionary of {image filename: saved path}
    """
    print(f"\nExtracting images from parquet file...")
    print(f"Parquet file: {parquet_path}")
    print(f"Output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas not installed")
        print("Please install: pip install pandas")
        return {}
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_path)
        print(f"Read {len(df)} records")
        print(f"Column names: {list(df.columns)}")
        
        image_map = {}
        
        # Find column containing images
        # Images are usually in 'image', 'image_bytes', 'image_path' and other columns
        image_column = None
        for col in ['image', 'image_bytes', 'image_path', 'image_data', 'data']:
            if col in df.columns:
                image_column = col
                break
        
        if not image_column:
            print("Warning: Image column not found, trying all columns...")
            # Try all columns
            for col in df.columns:
                sample = df[col].iloc[0] if len(df) > 0 else None
                if isinstance(sample, bytes) or (isinstance(sample, str) and len(sample) > 100):
                    image_column = col
                    print(f"Found possible image column: {col}")
                    break
        
        if not image_column:
            print("Error: Cannot find image column")
            return {}
        
        print(f"Using column: {image_column}")
        
        # Extract images
        for idx, row in df.iterrows():
            image_data = row[image_column]
            
            # Use index number as filename (match format in label.json)
            # Filename format in label.json is "0.jpeg", "1.jpeg", etc.
            filename = f"{idx}.jpeg"
            
            # Save image
            image_path = os.path.join(output_dir, filename)
            
            if isinstance(image_data, bytes):
                # Save byte data directly
                with open(image_path, 'wb') as f:
                    f.write(image_data)
            elif isinstance(image_data, str):
                # May be base64 encoded or path
                if image_data.startswith('data:image') or len(image_data) > 1000:
                    # May be base64
                    try:
                        import base64
                        # Remove data:image/xxx;base64, prefix
                        if ',' in image_data:
                            image_data = image_data.split(',')[1]
                        image_bytes = base64.b64decode(image_data)
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                    except Exception as e:
                        print(f"Warning: Cannot decode base64 image {filename}: {e}")
                        continue
                else:
                    # May be path, skip
                    print(f"Warning: Column {image_column} contains path instead of image data: {filename}")
                    continue
            else:
                print(f"Warning: Unknown image data type: {type(image_data)}")
                continue
            
            image_map[filename] = image_path
        
        print(f"✓ Extracted {len(image_map)} images")
        return image_map
        
    except Exception as e:
        print(f"Failed to extract images: {e}")
        import traceback
        traceback.print_exc()
        return {}


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


def find_image_file(image_filename: str, image_map: Dict[str, str]) -> Optional[str]:
    """Find image file in image map"""
    # Exact match
    if image_filename in image_map:
        return image_map[image_filename]
    
    # Filename (without extension) match
    image_stem = Path(image_filename).stem
    for stored_filename, path in image_map.items():
        if Path(stored_filename).stem == image_stem:
            return path
    
    # Filename contains relationship (case-insensitive)
    image_lower = image_filename.lower()
    for stored_filename, path in image_map.items():
        if image_lower in stored_filename.lower() or stored_filename.lower() in image_lower:
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
        except Exception as e:
            fail_count += 1
            failed_files.append(image_filename)
    
    return success_count, fail_count, failed_files


def process_nanonets_kie():
    """Main processing workflow"""
    print("="*60)
    print("Nanonets-KIE Dataset Processing Script")
    print("="*60)
    
    # 1. Download dataset
    print("\nStep 1: Download Nanonets-KIE dataset")
    dataset_dir = download_huggingface_dataset(NANONETS_KIE_REPO, 'Nanonets-KIE')
    if not dataset_dir:
        print("Download failed, exiting")
        return
    
    # 2. Find parquet files
    print("\nStep 2: Find parquet files")
    parquet_files = find_parquet_files(dataset_dir)
    if not parquet_files:
        print("No parquet files found, exiting")
        return
    
    # 3. Extract images from parquet files
    print("\nStep 3: Extract images from parquet files")
    extract_dir = os.path.join(DATA_SOURCE_DIR, 'Nanonets-KIE_images')
    os.makedirs(extract_dir, exist_ok=True)
    
    all_image_map = {}
    for parquet_file in parquet_files:
        print(f"\nProcessing file: {parquet_file}")
        image_map = extract_images_from_parquet(parquet_file, extract_dir)
        all_image_map.update(image_map)
    
    if not all_image_map:
        print("No images extracted, exiting")
        return
    
    print(f"\nTotal extracted {len(all_image_map)} images")
    
    # 4. Process Tax-Compliant category
    print("\nStep 4: Copy images to Tax-Compliant category based on label.json")
    print("="*60)
    
    category_dir = os.path.join(DATASETS_ROOT, TARGET_CATEGORY)
    label_path = os.path.join(category_dir, 'label.json')
    images_dir = os.path.join(category_dir, 'images')
    
    if not os.path.exists(label_path):
        print(f"Error: {TARGET_CATEGORY} label.json not found: {label_path}")
        return
    
    # Load label.json
    label_data = load_label_json(label_path)
    if not label_data:
        print(f"Error: {TARGET_CATEGORY} label.json is empty")
        return
    
    # Extract image filenames
    image_filenames = extract_image_filenames(label_data)
    print(f"\n{TARGET_CATEGORY}: Found {len(image_filenames)} image filenames")
    
    # Copy images
    success, fail, failed_files = copy_images_for_category(
        TARGET_CATEGORY, image_filenames, all_image_map, images_dir
    )
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total: Success {success}")
    
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
    process_nanonets_kie()
