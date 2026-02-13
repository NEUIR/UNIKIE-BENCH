#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional

# Configuration
HW_FORMS_REPO = 'ift/handwriting_forms'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'
# Use mirror site
USE_MIRROR = True  # Use hf-mirror.com mirror

# Hw-Forms corresponds to category
TARGET_CATEGORY = 'Postal-Label'


def download_huggingface_dataset(repo_id: str, target_dir: str) -> Optional[str]:
    """Download specified parquet file from HuggingFace"""
    target_path = os.path.join(DATA_SOURCE_DIR, target_dir)
    
    # File path: data/test-00000-of-00001-49a9864a2c204eab.parquet
    parquet_filename = 'data/test-00000-of-00001-49a9864a2c204eab.parquet'
    # After download, file will be saved in target_path/data/ directory
    parquet_path = os.path.join(target_path, parquet_filename)
    
    if os.path.exists(parquet_path):
        print(f"Parquet file already exists: {parquet_path}")
        print("Skipping download. Delete the file to re-download.")
        return target_path
    
    print(f"Starting HuggingFace dataset download: {repo_id}")
    print(f"Target file: {parquet_filename}")
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
    
    # Use HfApi to download specified parquet file
    print("\nDownloading using HfApi...")
    try:
        # If using mirror, set endpoint
        if USE_MIRROR:
            api = HfApi(endpoint='https://hf-mirror.com')
            print("Using mirror site: hf-mirror.com")
        else:
            api = HfApi()
        
        # List all files first to see actual file structure
        print("Listing dataset files...")
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        print(f"Found {len(files)} files")
        print("First 20 files:")
        for f in files[:20]:
            print(f"  - {f}")
        
        # Find parquet files
        parquet_files = [f for f in files if f.endswith('.parquet')]
        print(f"\nFound {len(parquet_files)} parquet files:")
        for f in parquet_files:
            print(f"  - {f}")
        
        # Download specified parquet file
        if parquet_filename in files:
            print(f"\nDownloading: {parquet_filename}")
            downloaded_path = api.hf_hub_download(
                repo_id=repo_id,
                filename=parquet_filename,
                repo_type="dataset",
                local_dir=target_path,
                resume_download=True
            )
            print(f"  ✓ {parquet_filename}")
        else:
            # If not found, try downloading first parquet file
            if parquet_files:
                print(f"\nSpecified file not found, downloading first parquet file: {parquet_files[0]}")
                downloaded_path = api.hf_hub_download(
                    repo_id=repo_id,
                    filename=parquet_files[0],
                    repo_type="dataset",
                    local_dir=target_path,
                    resume_download=True
                )
                parquet_filename = parquet_files[0]
                parquet_path = os.path.join(target_path, parquet_filename)
                print(f"  ✓ {parquet_files[0]}")
            else:
                print("Error: No parquet files found")
                return None
        
        # Check if download succeeded
        if os.path.exists(parquet_path):
            print(f"✓ Download completed")
            return target_path
        else:
            print("Download failed: file does not exist")
            return None
            
    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        print("\nSuggestions:")
        print("1. Check network connection")
        print("2. Try manually accessing: https://hf-mirror.com/datasets/" + repo_id)
        print("3. Manually download and place in: " + target_path)
        return None


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
        
        # Extract images
        for idx, row in df.iterrows():
            image_data = row['image']
            
            # Determine filename - Postal-Label uses simple numeric index + .png
            # Format in label.json is "0.png", "1.png", "2.png", etc.
            filename = f"{idx}.png"
            
            # Save image
            image_path = os.path.join(output_dir, filename)
            
            # Handle different types of image data
            image_bytes = None
            
            if isinstance(image_data, bytes):
                # Direct byte data
                image_bytes = image_data
            elif isinstance(image_data, dict):
                # If dict, try to extract image data
                # Possible structures: {'bytes': b'...'} or {'path': '...'} or others
                if 'bytes' in image_data:
                    image_bytes = image_data['bytes']
                elif 'data' in image_data:
                    image_bytes = image_data['data']
                elif 'image' in image_data:
                    image_bytes = image_data['image']
                else:
                    # Print dict keys for debugging
                    print(f"Warning: Image data is dict, keys: {list(image_data.keys())}, filename: {filename}")
                    # Try to get first value
                    if len(image_data) > 0:
                        first_value = list(image_data.values())[0]
                        if isinstance(first_value, bytes):
                            image_bytes = first_value
                        else:
                            print(f"Warning: Cannot extract image data from dict: {filename}")
                            continue
                    else:
                        print(f"Warning: Dict is empty: {filename}")
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
                    except Exception as e:
                        print(f"Warning: Cannot decode base64 image {filename}: {e}")
                        continue
                else:
                    print(f"Warning: Column 'image' contains path instead of image data: {filename}")
                    continue
            else:
                print(f"Warning: Unknown image data type: {type(image_data)}")
                continue
            
            # Save image byte data
            if image_bytes:
                try:
                    with open(image_path, 'wb') as f:
                        f.write(image_bytes)
                    image_map[filename] = image_path
                except Exception as e:
                    print(f"Warning: Failed to save image {filename}: {e}")
                    continue
            
            image_map[filename] = image_path
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{len(df)} samples")
        
        print(f"\n✓ Extracted {len(image_map)} images")
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


def process_hw_forms():
    """Main processing workflow"""
    print("="*60)
    print("Hw-Forms Dataset Processing Script")
    print("="*60)
    
    # 1. Download parquet file
    print("\nStep 1: Downloading Hw-Forms parquet file")
    dataset_dir = download_huggingface_dataset(HW_FORMS_REPO, 'Hw-Forms')
    if not dataset_dir:
        print("Download failed, exiting")
        return
    
    # 2. Extract images from parquet file
    print("\nStep 2: Extracting images from parquet file")
    # Find downloaded parquet file
    parquet_files = list(Path(dataset_dir).rglob('*.parquet'))
    if not parquet_files:
        print(f"Error: Parquet file not found: {dataset_dir}")
        return
    
    parquet_path = str(parquet_files[0])
    print(f"Using parquet file: {parquet_path}")
    
    if not os.path.exists(parquet_path):
        print(f"Error: Parquet file not found: {parquet_path}")
        return
    
    extract_dir = os.path.join(DATA_SOURCE_DIR, 'Hw-Forms_images')
    os.makedirs(extract_dir, exist_ok=True)
    
    all_image_map = extract_images_from_parquet(parquet_path, extract_dir)
    
    if not all_image_map:
        print("No images extracted, exiting")
        return
    
    print(f"\nTotal extracted {len(all_image_map)} images")
    
    # 3. Process Postal-Label category
    print("\nStep 3: Copying images to Postal-Label category based on label.json")
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
        TARGET_CATEGORY, image_filenames, all_image_map, images_dir
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
    process_hw_forms()
