"""Generate a Mindee-like dataschema.json per file for each dataset in datasets/.

For a fair comparison of Mindee API performance across other models, we just add the name of each field
as the name and title in the dataschema, without any additional guidelines or descriptions.
The field type is always string, and all nested structures are flattened to a single level of nesting (with dotted titles
indicating the original hierarchy), because the Mindee API only supports one level of nesting.
"""

import json
import re
from pathlib import Path
from typing import Any

from unidecode import unidecode


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = REPO_ROOT / "datasets"

# Those names are reserved keywords in the Mindee API and would cause inference errors if used as-is.
FORBIDDEN_NAMES = frozenset({"fields", "items", "value", "values", "item", "field"})
FORBIDDEN_SUFFIX = "_"


def to_snake_case(name: str) -> str:
    """Convert a field name to lowercase snake_case, transliterating Unicode to ASCII."""
    # Transliterate non-ASCII characters (e.g. Chinese, accented) to ASCII
    s = unidecode(name)
    # Insert underscore before uppercase letters preceded by lowercase
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    # Replace any non-alphanumeric characters with underscores
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    return s.strip("_").lower()


def _sanitize_name(name: str) -> str:
    """Suffix names that collide with reserved Mindee API keywords and truncate to 125 chars."""
    if name in FORBIDDEN_NAMES:
        name = name + FORBIDDEN_SUFFIX
    return name[:125]


def _sanitize_title(title: str) -> str:
    """Suffix title components that collide with reserved Mindee API keywords and truncate to 125 chars."""
    parts = title.split(".")
    sanitized = [p + FORBIDDEN_SUFFIX if p.lower() in FORBIDDEN_NAMES else p for p in parts]
    return ".".join(sanitized)[:125]


def _make_string_field(name: str, title: str, is_array: bool = False) -> dict:
    """Create a simple string field descriptor."""
    return {
        "name": _sanitize_name(name),
        "type": "string",
        "title": _sanitize_title(title),
        "is_array": is_array,
        "guidelines": "",
        "description": "",
        "nested_fields": [],
        "unique_values": False,
        "classification_values": []
    }


def _merge_list_keys(value_list: list[dict]) -> dict:
    """Merge keys from all dict items in a list, keeping first occurrence value."""
    merged = {}
    for item in value_list:
        if isinstance(item, dict):
            for k, v in item.items():
                if k not in merged:
                    merged[k] = v
    return merged


def _build_subfields(obj: dict) -> list[dict]:
    """Build nested_fields list, flattening any nested dicts/lists-of-dicts
    to prevent double nesting (API only supports one level of nesting).

    Flattened fields get a dotted title (e.g. 'sub.nm') so that
    run_inference can reconstruct the original nested structure.
    """
    fields = []
    for k, v in obj.items():
        if isinstance(v, dict):
            # Would create double nesting — flatten with prefix, dotted title
            for sub_k in v:
                fields.append(_make_string_field(
                    to_snake_case(f"{k}_{sub_k}"), f"{k}.{sub_k}"
                ))
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            # Would create double nesting — flatten with prefix, dotted title
            all_keys = _merge_list_keys(v)
            for sub_k in all_keys:
                fields.append(_make_string_field(
                    to_snake_case(f"{k}_{sub_k}"), f"{k}.{sub_k}"
                ))
        elif isinstance(v, list):
            # Array of scalars
            fields.append(_make_string_field(to_snake_case(k), k, is_array=True))
        else:
            # Simple scalar
            fields.append(_make_string_field(to_snake_case(k), k))
    return fields


def make_field(name: str, value: Any = None) -> dict:
    """Build a dataschema field descriptor, supporting nested objects and arrays.

    When value is provided, the structure is inferred:
    - dict -> nested_object with nested_fields (sub-nested objects are flattened)
    - list of dicts -> nested_object with is_array=True (sub-nested flattened)
    - list of strings -> string with is_array=True
    - otherwise -> string
    """
    snake = to_snake_case(name)

    if isinstance(value, dict):
        nested = _build_subfields(value)
        return {
            "name": _sanitize_name(snake),
            "type": "nested_object",
            "title": _sanitize_title(name),
            "is_array": False,
            "guidelines": "",
            "description": "",
            "nested_fields": nested,
            "unique_values": False,
            "classification_values": []
        }

    if isinstance(value, list):
        if value and isinstance(value[0], dict):
            merged = _merge_list_keys(value)
            nested = _build_subfields(merged)
            return {
                "name": _sanitize_name(snake),
                "type": "nested_object",
                "title": _sanitize_title(name),
                "is_array": True,
                "guidelines": "",
                "description": "",
                "nested_fields": nested,
                "unique_values": False,
                "classification_values": []
            }
        return _make_string_field(snake, name, is_array=True)

    return _make_string_field(snake, name)


def main() -> None:
    for dataset_dir in sorted(DATASETS_DIR.iterdir()):
        label_path = dataset_dir / "label.json"
        if not label_path.is_file():
            continue

        labels = json.loads(label_path.read_text(encoding="utf-8"))

        out_dir = dataset_dir / "dataschemas"
        out_dir.mkdir(exist_ok=True)

        count = 0
        for filename, entry in labels.items():
            schema = [make_field(key, value) for key, value in entry.items()]

            out_path = out_dir / f"{filename}.json"
            out_path.write_text(
                json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            count += 1

        print(f"{dataset_dir.name}: {count} files -> {out_dir}/")


if __name__ == "__main__":
    main()
