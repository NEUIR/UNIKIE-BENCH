"""Merge multi-page image subfolders in Commercial/images into single PDFs, so that mindee API can process them like a single document.

Each subfolder under datasets/Commercial/images/ is treated as a multi-page
document. The images inside are sorted by the first number found in their
filename and combined into a single PDF named <subfolder>.pdf.
"""

import re
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = REPO_ROOT / "datasets" / "Commercial" / "images"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def page_sort_key(filename: str) -> int:
    """Extract the first integer from *filename* for sorting."""
    m = re.search(r"(\d+)", filename)
    return int(m.group(1)) if m else 0


def merge_folder_to_pdf(folder: Path, output: Path) -> int:
    """Convert all images in *folder* into a single PDF at *output*.

    Returns the number of pages written.
    """
    pages = sorted(
        [f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS],
        key=page_sort_key,
    )
    if not pages:
        return 0

    images = [Image.open(folder / p).convert("RGB") for p in pages]
    try:
        images[0].save(output, save_all=True, append_images=images[1:])
    finally:
        for img in images:
            img.close()

    return len(pages)


def main() -> None:
    for entry in sorted(IMAGES_DIR.iterdir()):
        if not entry.is_dir():
            continue

        pdf_path = IMAGES_DIR / f"{entry.name}.pdf"
        n_pages = merge_folder_to_pdf(entry, pdf_path)
        if n_pages:
            print(f"{entry.name}: {n_pages} pages -> {entry.name}.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
