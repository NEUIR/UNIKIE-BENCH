#!/usr/bin/env python3

import os
import json
from pathlib import Path
from typing import Dict, Set, Optional

# Configuration
DOCILE_PDFS_DIR = './datasets_process/dataset_source/docile/pdfs'
DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

# DOCILE corresponds to category
TARGET_CATEGORY = 'Commercial'


def load_label_json(label_path: str) -> Dict:
    """Load label.json file"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def find_pdf_file(pdf_name: str, search_dir: str) -> Optional[str]:
    """Find corresponding PDF file in search directory"""
    search_path = Path(search_dir)
    
    # PDF filename (with extension)
    pdf_filename = f"{pdf_name}.pdf"
    
    # Strategy 1: Exact filename match (case-insensitive)
    for pdf_path in search_path.rglob('*.pdf'):
        if pdf_path.is_file():
            if pdf_path.name.lower() == pdf_filename.lower():
                return str(pdf_path)
    
    # Strategy 2: Filename (without extension) match (case-insensitive)
    pdf_name_lower = pdf_name.lower()
    for pdf_path in search_path.rglob('*.pdf'):
        if pdf_path.is_file():
            if pdf_path.stem.lower() == pdf_name_lower:
                return str(pdf_path)
    
    return None


def pdf_to_images(pdf_path: str, output_dir: str) -> int:
    """Convert PDF file to images, save to output directory
    Returns number of successfully converted images
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        try:
            from pdf2image import convert_from_bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            images = convert_from_bytes(pdf_bytes, dpi=200)
            if images:
                success_count = 0
                for idx, image in enumerate(images):
                    output_path = os.path.join(output_dir, f"page_{idx + 1}.jpg")
                    image.save(output_path, quality=95)
                    success_count += 1
                return success_count
            return 0
        except ImportError:
            return 0
    
    try:
        # Use convert_from_path
        try:
            images = convert_from_path(pdf_path, dpi=200)
        except:
            # If failed, try using convert_from_bytes
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            from pdf2image import convert_from_bytes
            images = convert_from_bytes(pdf_bytes, dpi=200)
        
        if images:
            success_count = 0
            for idx, image in enumerate(images):
                output_path = os.path.join(output_dir, f"page_{idx + 1}.jpg")
                image.save(output_path, quality=95)
                success_count += 1
            return success_count
        return 0
    except Exception:
        return 0


def process_docile():
    """Main processing workflow"""
    # 1. Check if PDFs directory exists
    if not os.path.exists(DOCILE_PDFS_DIR):
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
    
    # Extract PDF names (keys in label.json are PDF filenames without extension)
    # Only process PDFs that exist in label.json, not all PDFs
    # Filter out keys ending with image extensions (these are from other datasets)
    pdf_names = set(
        key for key in label_data.keys() 
        if not key.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    )
    
    # Create output directory
    os.makedirs(images_dir, exist_ok=True)
    
    success_count = 0
    
    # Try importing tqdm for progress bar
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
    
    # 3. Find corresponding PDF files based on keys in label.json and render
    pdf_names_list = sorted(pdf_names)
    iterator = tqdm(pdf_names_list, desc="Processing PDFs") if use_tqdm else pdf_names_list
    
    for pdf_name in iterator:
        # Find corresponding PDF file
        pdf_path = find_pdf_file(pdf_name, DOCILE_PDFS_DIR)
        
        if not pdf_path:
            # If corresponding PDF not found, skip
            continue
        
        # Create folder named after PDF name
        pdf_folder = os.path.join(images_dir, pdf_name)
        
        # If folder exists and contains images, skip
        if os.path.exists(pdf_folder) and os.path.isdir(pdf_folder):
            # Check if folder contains image files
            if any(f.endswith(('.jpg', '.jpeg', '.png')) for f in os.listdir(pdf_folder)):
                success_count += 1
                continue
        
        # Create folder
        os.makedirs(pdf_folder, exist_ok=True)
        
        # Convert PDF to images
        image_count = pdf_to_images(pdf_path, pdf_folder)
        if image_count > 0:
            success_count += 1
    
    # Output result
    print(f"Success: {success_count}")


if __name__ == '__main__':
    process_docile()
