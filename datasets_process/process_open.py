#!/usr/bin/env python3

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Set, Optional

SYN_DATA_REPO = 'sigdgsde2/UNIKIE-Open-Category'
DATASETS_ROOT = './datasets'

USE_MIRROR = True 

OPEN_CATEGORIES = [
    'Contract-cn',
    'Contract-en',
    'Form-cn',
    'Form-en',
    'invoice-cn',
    'invoice-en',
    'receipt-cn',
    'receipt-en'
]


def download_huggingface_dataset(repo_id: str, target_dir: str) -> Optional[str]:
    """Download dataset from HuggingFace directly to datasets directory"""
    # Check if any category directories already exist
    existing_categories = [cat for cat in OPEN_CATEGORIES if os.path.exists(os.path.join(DATASETS_ROOT, cat))]
    if existing_categories:
        print(f"Some categories already exist: {existing_categories}")
        print("Skipping download. Delete the directories to re-download.")
        return DATASETS_ROOT
    
    print(f"Starting HuggingFace dataset download: {repo_id}")
    print(f"Target: {DATASETS_ROOT}")
    
    os.makedirs(DATASETS_ROOT, exist_ok=True)
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        import sys
        print("Error: huggingface_hub not installed")
        print(f"Python path: {sys.executable}")
        print("Please install: pip install huggingface_hub")
        print(f"Detailed error: {e}")
        return None
    
    # Use snapshot_download to download dataset
    print("\nDownloading using snapshot_download...")
    try:
        # Download to a temporary directory first
        temp_dir = os.path.join(DATASETS_ROOT, '.temp_download')
        os.makedirs(temp_dir, exist_ok=True)
        
        # If using mirror, we need to handle it differently
        # Note: snapshot_download doesn't support mirror endpoint directly
        # So we'll use the default endpoint
        if USE_MIRROR:
            print("Warning: Mirror site not supported for snapshot_download, using default HuggingFace endpoint")
        
        print("Downloading dataset files...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=temp_dir,
            resume_download=True
        )
        
        # Check if download succeeded
        if not os.path.exists(temp_dir) or not any(Path(temp_dir).iterdir()):
            print("Download failed: directory is empty")
            return None
        
        # Find and move category directories, skipping outer folder if it exists
        print("\nProcessing downloaded files...")
        temp_path = Path(temp_dir)
        
        # Look for category directories
        # They might be directly in temp_dir or in a subdirectory (like UNIKIE-Syn-Data)
        source_dirs = []
        
        # Check if there's an outer folder (like UNIKIE-Syn-Data)
        for item in temp_path.iterdir():
            if item.is_dir():
                # Check if this directory contains our categories
                category_found = False
                for cat in OPEN_CATEGORIES:
                    if (item / cat).exists():
                        category_found = True
                        break
                
                if category_found:
                    # This is the outer folder, use it as source
                    source_dirs.append(item)
                    print(f"Found outer folder: {item.name}, will skip it and extract categories directly")
                elif item.name in OPEN_CATEGORIES:
                    # Category is directly in temp_dir
                    source_dirs.append(temp_path)
                    break
        
        if not source_dirs:
            # No outer folder found, categories should be directly in temp_dir
            source_dirs.append(temp_path)
        
        # Move category directories to final location
        moved_count = 0
        for source_dir in source_dirs:
            for category in OPEN_CATEGORIES:
                category_source = source_dir / category
                category_target = Path(DATASETS_ROOT) / category
                
                if category_source.exists() and category_source.is_dir():
                    if category_target.exists():
                        print(f"  ⚠ {category} already exists, skipping")
                        continue
                    
                    try:
                        shutil.move(str(category_source), str(category_target))
                        print(f"  ✓ Moved {category}")
                        moved_count += 1
                    except Exception as e:
                        print(f"  ✗ Failed to move {category}: {e}")
        
        # Clean up temp directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory")
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")
        
        if moved_count > 0:
            print(f"\n✓ Download completed, moved {moved_count} categories")
            return DATASETS_ROOT
        else:
            print("Download completed but no categories were moved")
            return None
            
    except Exception as e:
        print(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_open_category():
    """Main processing workflow"""
    print("="*60)
    print("Open Category Dataset Processing Script")
    print("="*60)
    
    # 1. Download dataset from HuggingFace directly to datasets directory
    print("\nStep 1: Downloading synthesized data from HuggingFace")
    print(f"Repository: {SYN_DATA_REPO}")
    print(f"Target: {DATASETS_ROOT}")
    
    result = download_huggingface_dataset(SYN_DATA_REPO, 'open-category')
    if not result:
        print("Download failed, exiting")
        return
    
    print(f"\nDataset downloaded to: {DATASETS_ROOT}")
    
    # 2. Verify categories
    print("\nStep 2: Verifying categories")
    print("="*60)
    
    total_success = 0
    total_failed = 0
    
    for category in OPEN_CATEGORIES:
        category_dir = Path(DATASETS_ROOT) / category
        
        if category_dir.exists():
            # Check for required files
            has_label = (category_dir / "label.json").exists()
            has_qa = (category_dir / "qa.jsonl").exists()
            has_images = (category_dir / "images").exists() and (category_dir / "images").is_dir()
            
            print(f"\n{category}:")
            if has_label:
                print(f"  ✓ label.json")
            else:
                print(f"  ✗ label.json missing")
            
            if has_qa:
                print(f"  ✓ qa.jsonl")
            else:
                print(f"  ✗ qa.jsonl missing")
            
            if has_images:
                print(f"  ✓ images/")
            else:
                print(f"  ⚠ images/ not found (optional)")
            
            if has_label and has_qa:
                print(f"  ✓ Category ready")
                total_success += 1
            else:
                print(f"  ✗ Category incomplete")
                total_failed += 1
        else:
            print(f"\n{category}: ✗ Directory not found")
            total_failed += 1
    
    # Summary
    print("\n" + "="*60)
    print("Processing Summary")
    print("="*60)
    print(f"Total categories: {len(OPEN_CATEGORIES)}")
    print(f"Successfully processed: {total_success}")
    print(f"Failed: {total_failed}")
    print(f"\n✓ Processing completed!")
    print(f"Data saved to: {DATASETS_ROOT}")


if __name__ == '__main__':
    process_open_category()
