#!/usr/bin/env python3

import os
import json
from pathlib import Path
from typing import Dict, List, Set, Optional
try:
    from tqdm import tqdm
except ImportError:
    # If tqdm is not available, use simple alternative
    def tqdm(iterable, desc=""):
        return iterable

# Configuration
DEEPFORM_DIR = './datasets_process/dataset_source/DeepForm/DeepForm'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# DeepForm corresponds to category
TARGET_CATEGORY = 'Advertisement'


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


def find_pdf_file(image_filename: str, search_dir: str) -> Optional[str]:
    """Find corresponding PDF file in search directory"""
    search_path = Path(search_dir)
    
    # Convert image filename to PDF filename (remove image extension, add .pdf)
    image_stem = Path(image_filename).stem
    pdf_filename = f"{image_stem}.pdf"
    
    # Strategy 1: Exact filename match (case-insensitive)
    for pdf_path in search_path.rglob('*.pdf'):
        if pdf_path.is_file():
            if pdf_path.name.lower() == pdf_filename.lower():
                return str(pdf_path)
    
    # Strategy 2: Filename (without extension) match (case-insensitive)
    image_stem_lower = image_stem.lower()
    for pdf_path in search_path.rglob('*.pdf'):
        if pdf_path.is_file():
            if pdf_path.stem.lower() == image_stem_lower:
                return str(pdf_path)
    
    # Strategy 3: Filename contains relationship (case-insensitive)
    for pdf_path in search_path.rglob('*.pdf'):
        if pdf_path.is_file():
            pdf_stem = pdf_path.stem.lower()
            if image_stem_lower in pdf_stem or pdf_stem in image_stem_lower:
                return str(pdf_path)
    
    return None


def pdf_to_image(pdf_path: str, output_path: str) -> bool:
    """Convert PDF file to image"""
    try:
        from pdf2image import convert_from_path
    except ImportError:
        try:
            from pdf2image import convert_from_bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)
            if images:
                images[0].save(output_path, quality=95)
                return True
            return False
        except ImportError:
            return False
    
    try:
        # Use convert_from_path
        try:
            images = convert_from_path(pdf_path, dpi=200, first_page=1, last_page=1)
        except:
            # If failed, try using convert_from_bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, dpi=200, first_page=1, last_page=1)
        
        if images:
            # Save first page, filename matches label.json
            images[0].save(output_path, quality=95)
            return True
        return False
    except Exception:
        return False


def process_deepform():
    """Main processing workflow"""
    # 1. Check if DeepForm directory exists
    if not os.path.exists(DEEPFORM_DIR):
        print(f"Error: DeepForm directory does not exist: {DEEPFORM_DIR}")
        return
    
    # 2. Load label.json
    category_dir = os.path.join(DATASETS_ROOT, TARGET_CATEGORY)
    label_path = os.path.join(category_dir, 'label.json')
    images_dir = os.path.join(category_dir, 'images')
    
    if not os.path.exists(label_path):
        print(f"Error: label.json not found: {label_path}")
        return
    
    label_data = load_label_json(label_path)
    if not label_data:
        print(f"Error: label.json is empty: {label_path}")
        return
    
    # Extract image filenames
    image_filenames = extract_image_filenames(label_data)
    
    if not image_filenames:
        print(f"Error: No image filenames found")
        return
    
    # Create output directory
    os.makedirs(images_dir, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    not_found_count = 0
    
    # Check if pdf2image is available
    try:
        from pdf2image import convert_from_path, convert_from_bytes
    except ImportError:
        print("Error: pdf2image not installed")
        print("Please install: pip install pdf2image")
        print("Note: Also need to install poppler-utils")
        return
    
    # 3. Process each image file
    image_filenames_sorted = sorted(image_filenames)
    for image_filename in tqdm(image_filenames_sorted, desc="Processing PDFs"):
        # Target image path (filename matches label.json)
        dest_image_path = os.path.join(images_dir, image_filename)
        
        # If target file exists, skip
        if os.path.exists(dest_image_path):
            success_count += 1
            continue
        
        # Find corresponding PDF file
        pdf_path = find_pdf_file(image_filename, DEEPFORM_DIR)
        
        if not pdf_path:
            not_found_count += 1
            continue
        
        # Convert PDF to image
        if pdf_to_image(pdf_path, dest_image_path):
            success_count += 1
        else:
            failed_count += 1
    
    # Output result
    print(f"Success: {success_count}")
    if not_found_count > 0:
        print(f"PDFs not found: {not_found_count}")
    if failed_count > 0:
        print(f"Conversion failed: {failed_count}")


if __name__ == '__main__':
    process_deepform()
