#!/usr/bin/env python3
import base64
import csv
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

from huggingface_hub import hf_hub_download

DATA_SOURCE_DIR = './datasets_process/dataset_source'
DATASETS_ROOT = './datasets'

SIBR_CATEGORIES = ['Accommodation', 'Medical-Services', 'Commercial']

_KIE_TSV_PARTS = ('kie', 'open_category', 'COLD_SIBR_400.tsv')


def _remote_dataset_id() -> str:
    """Resolve dataset repo id; prefer env so nothing sensitive is hard-coded."""
    env = os.environ.get('UNIKIE_KIE_SOURCE')
    if env and env.strip():
        return env.strip()
    return base64.b64decode('d3VsaXBjL0NDLU9DUg==').decode('ascii')


def _kie_tsv_filename() -> str:
    return '/'.join(_KIE_TSV_PARTS)


def download_kie_tsv() -> Optional[Path]:
    """Download TSV from Hugging Face (uses local cache when unchanged)."""
    try:
        print('Fetching KIE table file (uses HF cache when already present)...')
        p = hf_hub_download(
            repo_id=_remote_dataset_id(),
            filename=_kie_tsv_filename(),
            repo_type='dataset',
        )
        path = Path(p)
        print(f'✓ Table ready: {path}')
        return path
    except Exception as e:
        print(f'✗ Download failed: {e}')
        return None


def materialize_images_from_tsv(
    tsv_path: Path,
    extract_dir: str,
    needed_filenames: Set[str],
) -> bool:
    """Decode base64 image column for rows whose image_name is needed."""
    os.makedirs(extract_dir, exist_ok=True)

    csv.field_size_limit(sys.maxsize)
    written = 0
    skipped = 0
    missing_targets: Set[str] = set(needed_filenames)

    try:
        with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_ALL)
            required = {'image', 'image_name'}
            if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
                print('Error: TSV missing expected columns (need image, image_name).')
                return False

            for row in reader:
                name = row.get('image_name') or ''
                if name not in needed_filenames:
                    continue
                dest = os.path.join(extract_dir, name)
                if os.path.exists(dest):
                    skipped += 1
                    missing_targets.discard(name)
                    continue
                b64 = row.get('image') or ''
                try:
                    raw = base64.b64decode(b64, validate=False)
                except Exception:
                    print(f'Warning: could not decode image payload for {name}')
                    continue
                with open(dest, 'wb') as out:
                    out.write(raw)
                written += 1
                missing_targets.discard(name)

        print(f'✓ Decoded images: {written} new, {skipped} already on disk')
        if missing_targets:
            print(
                f'Warning: {len(missing_targets)} requested filenames '
                f'were not found in the table (showing up to 10): '
                f'{sorted(missing_targets)[:10]}'
            )
        return True
    except OSError as e:
        print(f'Error reading TSV: {e}')
        return False


def load_label_json(label_path: str) -> Dict:
    """Load label.json file"""
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f'Error: Cannot read {label_path}: {e}')
        return {}


def extract_image_filenames(label_data: Dict) -> Set[str]:
    """Extract all image filenames from label.json"""
    image_filenames = set()
    for key in label_data.keys():
        if any(key.lower().endswith(ext.lower()) for ext in ['.jpg', '.jpeg', '.png', '.bmp']):
            image_filenames.add(key)
    return image_filenames


def find_image_file(image_filename: str, search_dir: str) -> str:
    """Find image file in search directory"""
    search_path = Path(search_dir)

    for img_path in search_path.rglob(image_filename):
        if img_path.is_file():
            return str(img_path)

    image_stem = Path(image_filename).stem
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        for img_path in search_path.rglob(f"{image_stem}{ext}"):
            if img_path.is_file():
                return str(img_path)

    image_lower = image_filename.lower()
    for img_path in search_path.rglob('*'):
        if img_path.is_file():
            if image_lower in img_path.name.lower() or img_path.name.lower() in image_lower:
                return str(img_path)

    return None


def copy_images_for_category(
    category: str,
    image_filenames: Set[str],
    source_dir: str,
    dest_images_dir: str,
) -> tuple:
    """Copy images for a single category"""
    os.makedirs(dest_images_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    failed_files = []

    for image_filename in sorted(image_filenames):
        source_image_path = find_image_file(image_filename, source_dir)

        if not source_image_path:
            fail_count += 1
            failed_files.append(image_filename)
            continue

        dest_image_path = os.path.join(dest_images_dir, image_filename)

        if os.path.exists(dest_image_path):
            if os.path.getsize(dest_image_path) == os.path.getsize(source_image_path):
                success_count += 1
                continue

        try:
            shutil.copy2(source_image_path, dest_image_path)
            success_count += 1
        except Exception:
            fail_count += 1
            failed_files.append(image_filename)

    return success_count, fail_count, failed_files


def process_sibr():
    """Main processing workflow"""
    print('=' * 60)
    print('SIBR Dataset Processing Script')
    print('=' * 60)

    needed: Set[str] = set()
    for category in SIBR_CATEGORIES:
        label_path = os.path.join(DATASETS_ROOT, category, 'label.json')
        if not os.path.exists(label_path):
            continue
        label_data = load_label_json(label_path)
        needed |= extract_image_filenames(label_data)

    if not needed:
        print('No image keys found in label.json files; nothing to do.')
        return

    print(f'\nStep 1: Resolve {len(needed)} unique image filenames from label.json')
    tsv_path = download_kie_tsv()
    if not tsv_path:
        print('Download failed, exiting')
        return

    print('\nStep 2: Materialize images from table into workspace cache')
    extract_dir = os.path.join(DATA_SOURCE_DIR, 'SIBR_images')
    if not materialize_images_from_tsv(tsv_path, extract_dir, needed):
        print('Materialization failed, exiting')
        return

    print('\nStep 3: Copy images to category folders (split unchanged)')
    print('=' * 60)

    total_success = 0
    total_fail = 0
    all_failed_files: Dict[str, List[str]] = {}

    for category in SIBR_CATEGORIES:
        category_dir = os.path.join(DATASETS_ROOT, category)
        label_path = os.path.join(category_dir, 'label.json')
        images_dir = os.path.join(category_dir, 'images')

        if not os.path.exists(label_path):
            print(f'\nWarning: {category} label.json not found: {label_path}')
            continue

        label_data = load_label_json(label_path)
        if not label_data:
            print(f'\nWarning: {category} label.json is empty')
            continue

        image_filenames = extract_image_filenames(label_data)
        print(f'\n{category}: {len(image_filenames)} image filenames in labels')

        success, fail, failed_files = copy_images_for_category(
            category, image_filenames, extract_dir, images_dir
        )

        total_success += success
        total_fail += fail
        if failed_files:
            all_failed_files[category] = failed_files

    print('\n' + '=' * 60)
    print('Processing Summary')
    print('=' * 60)
    print(f'Total: Success {total_success}')

    if all_failed_files:
        print('\nFailed files:')
        for category, files in all_failed_files.items():
            print(f'\n{category} ({len(files)} files):')
            for f in files[:10]:
                print(f'  - {f}')
            if len(files) > 10:
                print(f'  ... and {len(files) - 10} more files')

    print('\n✓ Processing completed!')
    print('Images copied to respective category images folders')
    print(f'Decoded cache retained at: {extract_dir}')


if __name__ == '__main__':
    process_sibr()
