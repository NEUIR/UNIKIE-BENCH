"""Run Mindee API inference on all datasets and store results in label.json format."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from mindee import BytesInput, ClientV2, InferenceParameters, InferenceResponse
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}

FORBIDDEN_SUFFIX = "_"


def _unsanitize_title(title: str) -> str:
    """Strip the forbidden-name suffix added by generate_dataschemas from title components."""
    parts = title.split(".")
    cleaned = [p[:-len(FORBIDDEN_SUFFIX)] if p.endswith(FORBIDDEN_SUFFIX) else p for p in parts]
    return ".".join(cleaned)


def discover_datasets(base_path: Path) -> list[str]:
    """Discover datasets that have both images/ and dataschemas/ with content."""
    datasets = []
    for d in sorted(base_path.iterdir()):
        if not d.is_dir():
            continue
        images_dir = d / "images"
        schemas_dir = d / "dataschemas"
        if images_dir.exists() and schemas_dir.exists() and any(schemas_dir.iterdir()):
            datasets.append(d.name)
    return datasets


def find_image_for_schema(schema_file: Path, images_dir: Path) -> Path | None:
    """Find the image file corresponding to a dataschema file.

    Schema naming conventions:
    - 'hot-0300.jpg.json' -> image 'hot-0300.jpg'
    - '009bdcdec5c04cd3b2e31555.json' -> image '009bdcdec5c04cd3b2e31555.pdf'
    """
    # Remove .json suffix to get the candidate image name
    candidate = schema_file.stem  # e.g. 'hot-0300.jpg' or '009bdcdec5c04cd3b2e31555'

    # If candidate already has an image extension, look for it directly
    candidate_path = images_dir / candidate
    if candidate_path.is_file():
        return candidate_path

    # Otherwise try common image extensions
    for ext in IMAGE_EXTENSIONS:
        test_path = images_dir / f"{candidate}{ext}"
        if test_path.is_file():
            return test_path

    return None


def get_label_key(image_path: Path) -> str:
    """Derive the label key from the image path, matching label.json conventions.

    - JPG/image files: key includes extension (e.g. 'hot-0300.jpg')
    - PDF files: key is stem only (e.g. '009bdcdec5c04cd3b2e31555')
    """
    if image_path.suffix.lower() == ".pdf":
        return image_path.stem
    return image_path.name


def build_name_to_title(fields: list[dict]) -> dict:
    """Build a name -> (title, nested_mapping) mapping from the dataschema fields."""
    mapping = {}
    for f in fields:
        nested_map = {}
        if f.get("nested_fields"):
            nested_map = build_name_to_title(f["nested_fields"])
        mapping[f["name"]] = {"title": _unsanitize_title(f["title"]), "nested": nested_map}
    return mapping


def _unwrap(data: object) -> object:
    """Recursively strip confidence/locations wrappers, unwrap fields/items/value."""
    if isinstance(data, list):
        return [_unwrap(i) for i in data]
    if not isinstance(data, dict):
        return data
    if "fields" in data and isinstance(data["fields"], dict):
        return {k: _unwrap(v) for k, v in data["fields"].items()}
    if "items" in data and isinstance(data["items"], list):
        return [_unwrap(i) for i in data["items"]]
    if "value" in data and ("confidence" in data or "locations" in data):
        return data["value"] or ""
    return {k: _unwrap(v) for k, v in data.items()}


def extract_prediction_fields(inference_result: dict, field_mapping: dict) -> dict:
    """Extract field title -> value mapping from the Mindee inference result."""
    result = {}
    for field_name, raw in inference_result.items():
        info = field_mapping.get(field_name, {"title": field_name, "nested": {}})
        title, nested_map = info["title"], info["nested"]
        field_data = _unwrap(raw)

        if isinstance(field_data, list) and field_data and isinstance(field_data[0], dict):
            result[title] = [
                {nested_map.get(k, {"title": k})["title"]: v for k, v in item.items()}
                for item in field_data
            ]
        elif isinstance(field_data, dict) and "value" not in field_data:
            result[title] = {
                nested_map.get(k, {"title": k})["title"]: v for k, v in field_data.items()
            }
        elif isinstance(field_data, dict) and "value" in field_data:
            result[title] = field_data.get("value") or ""
        else:
            result[title] = field_data
    return result


def _unflatten_dotted(obj: dict, label_template) -> dict:
    """Convert keys with dots (e.g. 'sub.nm') back into nested dicts/lists.

    Uses label_template to determine whether to reconstruct as dict or list.
    """
    result = {}
    nested_groups = {}  # parent -> {child: value}

    for key, value in obj.items():
        if "." in str(key):
            parent, child = str(key).split(".", 1)
            if parent not in nested_groups:
                nested_groups[parent] = {}
            nested_groups[parent][child] = value
        else:
            result[key] = value

    for parent, children in nested_groups.items():
        if isinstance(label_template, dict):
            label_val = label_template.get(parent)
            if isinstance(label_val, list):
                result[parent] = [children]
            else:
                result[parent] = children
        else:
            result[parent] = children

    return result


def reconstruct_nested(prediction: dict, label_entry: dict) -> dict:
    """Reconstruct flattened dotted keys back into the original nested structure.

    Uses label_entry from label.json to determine list vs dict for each field.
    """
    result = {}
    for key, value in prediction.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            # Array of objects — unflatten each row
            label_val = label_entry.get(key, [])
            if isinstance(label_val, list) and label_val and isinstance(label_val[0], dict):
                sample = label_val[0]
            elif isinstance(label_val, dict):
                sample = label_val
            else:
                sample = {}
            result[key] = [_unflatten_dotted(item, sample) for item in value]
        elif isinstance(value, dict):
            # Nested object — unflatten
            label_val = label_entry.get(key, {})
            result[key] = _unflatten_dotted(value, label_val)
        else:
            result[key] = value
    return result


def process_dataset(
    dataset_name: str,
    base_path: Path,
    mindee_client: ClientV2,
    model_id: str,
    output_path: Path,
    confidence: bool = False,
) -> list[dict]:
    """Process a single dataset: run inference on each (image, dataschema) pair."""
    dataset_path = base_path / dataset_name
    images_dir = dataset_path / "images"
    schemas_dir = dataset_path / "dataschemas"

    schema_files = sorted(schemas_dir.glob("*.json"))
    if not schema_files:
        logger.warning(f"No dataschemas found for {dataset_name}, skipping")
        return []

    # Load label.json for structure reference (used for nested reconstruction)
    label_path = dataset_path / "label.json"
    labels_structure = json.loads(label_path.read_text()) if label_path.exists() else {}

    entries = []
    failed = []

    for schema_file in tqdm(schema_files, desc=f"  {dataset_name}"):
        image_path = find_image_for_schema(schema_file, images_dir)
        if image_path is None:
            logger.warning(f"No image found for schema {schema_file.name}")
            failed.append(schema_file.name)
            continue

        label_key = get_label_key(image_path)

        # Load per-image dataschema
        fields = json.loads(schema_file.read_text())
        data_schema = json.dumps({"replace": {"fields": fields}})
                
        params = InferenceParameters(
            model_id=model_id,
            data_schema=data_schema,
            confidence=confidence,
        )
        response = mindee_client.enqueue_and_get_result(
            response_type=InferenceResponse,
            input_source=BytesInput(image_path.read_bytes(), filename=image_path.name),
            params=params,
        )
        inference = response._raw_http["inference"]["result"]["fields"]

        # Build recursive name -> title mapping from the dataschema
        field_mapping = build_name_to_title(fields)
        prediction = extract_prediction_fields(inference, field_mapping)

        # Reconstruct flattened dotted keys back into original nested structure
        label_entry = labels_structure.get(label_key, {})
        prediction = reconstruct_nested(prediction, label_entry)

        entries.append({
            "dataset": dataset_name,
            "url": f"images/{label_key}",
            "model_result": prediction,
        })
        logger.info(f"  ✓ {label_key}")

    # Write per-dataset JSONL output
    mode_dir = "confidence" if confidence else "base"
    out_file = output_path / dataset_name / mode_dir / "predictions.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(
        f"{dataset_name}: {len(entries)}/{len(schema_files)} succeeded, "
        f"{len(failed)} failed -> {out_file}"
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Mindee API inference on all datasets and store results as predictions.json"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("MINDEE_API_KEY"),
        help="Mindee API key (default: $MINDEE_API_KEY)",
    )
    parser.add_argument("--base-url", default="https://api-v2.mindee.net", help="API base URL")
    parser.add_argument("--datasets-path", type=Path, default=Path("datasets"), help="Datasets dir")
    parser.add_argument("--output", type=Path, default=Path("out"), help="Output directory")
    parser.add_argument(
        "--confidence", action="store_true", default=False,
        help="Enable confidence scores in inference results.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Model ID (same org as api-key); dataschema is overridden per image.",
    )
    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set $MINDEE_API_KEY)")

    os.environ["MINDEE_V2_BASE_URL"] = f"{args.base_url}/v2"
    mindee_client = ClientV2(args.api_key)

    datasets = discover_datasets(args.datasets_path)
    logger.info(f"Found {len(datasets)} datasets: {', '.join(datasets)}")

    if not datasets:
        logger.error("No datasets found")
        sys.exit(1)

    total = 0
    for ds in datasets:
        logger.info(f"--- {ds} ---")
        entries = process_dataset(ds, args.datasets_path, mindee_client, args.model_id, args.output, args.confidence)
        total += len(entries)

    logger.info(f"Wrote {total} predictions across {len(datasets)} datasets")
    sys.exit(0 if total else 1)


if __name__ == "__main__":
    main()
