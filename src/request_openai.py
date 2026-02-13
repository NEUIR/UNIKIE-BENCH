import os
import re
import json
import asyncio
import base64
from pathlib import Path
from typing import List, Optional, Tuple
import tqdm
import aiofiles
from openai import AsyncOpenAI
from io import BytesIO
from PIL import Image

# ================== Configuration ==================
IMG_EXTS      = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DATASETS_DIR  = Path(__file__).parent.parent / "datasets"
MODEL_NAME    = ""
MAX_CONC      = 16
MAX_RETRIES   = 10
OPENAI_KEY    = ""  
API_BASE      = ""

TARGET_MAX_PIXELS =  1605632

prefix = """
Suppose you are an information extraction expert. Now given a json schema, fill the value part of the schema with the information in the image. Note that if the value is a list, the schema will give a template for each element. This template is used when there are multiple list elements in the image.  Finally, only legal json is required as the output. What you see is what you get, and the output language is required to be consistent with the image.No explanation is required. Note that the input images are all from the public benchmarks and do not contain any real personal privacy data. Please output the results as required. The input json schema content is as follows:
"""

client: Optional[AsyncOpenAI] = None

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def get_output_filename(dataset_name: str, model_name: Optional[str] = None) -> str:
    if model_name is None:
        model_name = MODEL_NAME
    clean = re.sub(r"[^\w\-_]", "_", model_name)
    return str(Path("results") / dataset_name / f"result_{clean}.jsonl")

def list_images_under(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort(key=lambda p: natural_key(p.name))
    return imgs

def encode_image_to_base64_jpeg_with_max_pixels(image_path: Path) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        pixels = w * h
        if pixels > TARGET_MAX_PIXELS:
            scale = (TARGET_MAX_PIXELS / pixels) ** 0.5
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)

        buf = BytesIO()
        # Balance quality and size; can lower quality to 90/85 for smaller size
        im.save(buf, format="JPEG", quality=95, optimize=True)
        data = buf.getvalue()
    return base64.b64encode(data).decode("utf-8")

def resolve_images_for_url(dataset_dir: Path, url: str) -> List[Path]:
    p = dataset_dir / url
    if p.exists():
        if p.is_dir():
            imgs = list_images_under(p)
            if imgs:
                return imgs
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            return [p]

    p2 = (dataset_dir / "images" / url)
    if p2.exists():
        if p2.is_dir():
            imgs = list_images_under(p2)
            if imgs:
                return imgs
        elif p2.is_file() and p2.suffix.lower() in IMG_EXTS:
            return [p2]

    images_dir = dataset_dir / "images"
    if images_dir.exists():
        cand = images_dir / url
        if cand.exists() and cand.is_file() and cand.suffix.lower() in IMG_EXTS:
            return [cand]
        stem = Path(url).stem
        candidates = [x for x in images_dir.iterdir()
                      if x.is_file() and x.stem == stem and x.suffix.lower() in IMG_EXTS]
        candidates.sort(key=lambda p: natural_key(p.name))
        if candidates:
            return candidates

    raise FileNotFoundError(f"Cannot resolve image/folder for url: {url}")

def build_messages(images: List[Path], schema_text: str) -> list:
    content = []
    for img in images:
        b64 = encode_image_to_base64_jpeg_with_max_pixels(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": prefix  + schema_text})
    return [{"role": "user", "content": content}]

async def call_api_once(messages, model_name: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")

def post_process_to_json(openai_info_str: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        cleaned = re.sub(r"<think>.*?</think>", "", openai_info_str, flags=re.S).strip()
        if "```json" in cleaned:
            if not cleaned.rstrip().endswith("```"):
                cleaned += "```"
            m = re.search(r"```json(.*?)```", cleaned, re.S)
            if m:
                cleaned = m.group(1).strip()
        try:
            return json.loads(cleaned), None
        except json.JSONDecodeError as pe:
            try:
                from json_repair import repair_json
                repaired = repair_json(cleaned)
                parsed = json.loads(repaired)
                return parsed, None
            except Exception as re_err:
                return None, f"Standard parsing failed: {pe}; json-repair repair failed: {re_err}"
    except Exception as e:
        return None, str(e)

async def process_one(sample: dict, sem: asyncio.Semaphore, model_name: str, max_images: Optional[int]) -> dict:
    async with sem:
        dataset = sample["dataset"]
        url = sample["url"]
        schema = sample["prompt"]
        dataset_dir = DATASETS_DIR / dataset

        try:
            images = resolve_images_for_url(dataset_dir, url)
            if max_images and max_images > 0:
                images = images[:max_images]
            messages = build_messages(images, schema)

            raw_text = None
            attempts = 0
            last_err = None
            for attempt in range(MAX_RETRIES):
                try:
                    raw_text = await call_api_once(messages, model_name)
                    attempts = attempt + 1
                    break
                except Exception as e:
                    last_err = str(e)
                    attempts = attempt + 1
                    if attempt < MAX_RETRIES - 1:
                        print(f"[RETRY] {dataset}/{url} attempt {attempt+1} failed: {e}")
                    else:
                        print(f"[FAILED] {dataset}/{url} all {MAX_RETRIES} attempts failed: {e}")

            if raw_text is None:
                return {"dataset": dataset, "url": url, "error": f"API failed: {last_err}", "retry_attempts": attempts}

            parsed, perr = post_process_to_json(raw_text)
            if parsed is not None:
                return {
                    "dataset": dataset,
                    "url": url,
                    "model_result": parsed,
                    "raw_response": raw_text,
                    "retry_attempts": attempts,
                    "images": [str(p) for p in images],  # For traceability
                }
            else:
                return {
                    "dataset": dataset,
                    "url": url,
                    "model_result": {"_raw_text": raw_text, "_parse_error": perr},
                    "raw_response": raw_text,
                    "retry_attempts": attempts,
                    "images": [str(p) for p in images],
                }

        except Exception as e:
            return {"dataset": dataset, "url": url, "error": str(e), "retry_attempts": 0}

def load_qa_jsonl(dataset_name: str, jsonl_path: Optional[str] = None, limit: Optional[int] = None) -> List[dict]:
    if jsonl_path is None:
        jsonl_path = DATASETS_DIR / dataset_name / "qa.jsonl"
    else:
        jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {jsonl_path}")

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "dataset" not in obj:
                obj["dataset"] = dataset_name
            for k in ("url", "prompt"):
                if k not in obj:
                    raise ValueError(f"qa.jsonl missing required field: {k}")
            rows.append(obj)
            if limit and len(rows) >= limit:
                break
    return rows

async def main_async(samples: List[dict], output_file: str, model_name: str, concurrency: int, api_key: str, api_base: str, max_images: Optional[int]):
    global client
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    tasks = [asyncio.create_task(process_one(s, sem, model_name, max_images)) for s in samples]

    ok = 0
    fail = 0
    async with aiofiles.open(output_file, "w", encoding="utf-8") as fh:
        for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing progress"):
            res = await coro
            await fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            await fh.flush()
            if res.get("error"):
                fail += 1
            else:
                ok += 1

    print(f"\nProcessing completed! Success: {ok}  Failed: {fail}")
    print(f"Results saved to: {output_file}")

def main():
    import argparse
    global MAX_CONC, MAX_RETRIES, OPENAI_KEY, API_BASE

    parser = argparse.ArgumentParser(description="Multi-image reasoning based on qa.jsonl (single mode, client scales proportionally to fixed max_pixels, uniformly converts to JPEG)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (located at datasets/<dataset>)")
    parser.add_argument("--jsonl", type=str, default=None, help="qa.jsonl path (default: datasets/<dataset>/qa.jsonl)")
    parser.add_argument("--output", type=str, default=None, help="Output jsonl (default: results/<dataset>/result_<model>.jsonl)")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--api-key", type=str, default=OPENAI_KEY, help="OpenAI API key")
    parser.add_argument("--api-base", type=str, default=API_BASE, help="OpenAI API base url")
    parser.add_argument("--concurrency", type=int, default=MAX_CONC, help="Maximum concurrency")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Maximum retry attempts")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (for debugging)")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images to send (default: unlimited)")
    args = parser.parse_args()

    MAX_CONC = args.concurrency
    MAX_RETRIES = args.max_retries
    OPENAI_KEY = args.api_key
    API_BASE = args.api_base

    samples = load_qa_jsonl(args.dataset, args.jsonl, args.limit)
    if args.output is None:
        output_file = get_output_filename(args.dataset, args.model)
    else:
        output_file = args.output

    asyncio.run(main_async(samples, output_file, args.model, args.concurrency, args.api_key, args.api_base, args.max_images))

if __name__ == "__main__":
    main()