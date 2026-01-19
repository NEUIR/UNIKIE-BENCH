
import json
import argparse
from pathlib import Path
from collections import defaultdict

import kie_evaluator

DATASETS_DIR = Path(__file__).parent.parent / "datasets"


def normalize_func(text, **kwargs):
    """Text normalization function"""
    halfwidth_text = kie_evaluator.fullwidth_to_halfwidth(str(text))
    cleaned_text = kie_evaluator.remove_unnecessary_spaces(halfwidth_text)
    return cleaned_text


def extract_image_name(url: str) -> str:
    if url.startswith("images/"):
        return url[7:]
    return url


def load_predictions(pred_jsonl_path: str) -> dict:
    predictions = defaultdict(dict)
    
    print(f"Loading predictions: {pred_jsonl_path}")
    
    with open(pred_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                
                # Get dataset name and URL
                dataset = data.get("dataset", "unknown")
                url = data.get("url", "")
                
                if not url:
                    print(f"Warning: Line {line_num} missing url field, skipping")
                    continue
                
                image_name = extract_image_name(url)
                model_result = data.get("model_result")
            
                if "error" in data:
                    print(f"Warning: {dataset}/{image_name} has error: {data['error']}")
                
                if model_result is None:
                    print(f"Warning: {dataset}/{image_name} has no prediction result")
                    continue
    
                if "_parse_error" in model_result:
                    print(f"Warning: {dataset}/{image_name} JSON parsing failed: {model_result.get('_parse_error')}")
                    continue
                
                predictions[dataset][image_name] = model_result
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} JSON parsing failed: {e}")
                continue
            except Exception as e:
                print(f"Error: Line {line_num} processing failed: {e}")
                continue
    
    # Statistics
    total_samples = sum(len(preds) for preds in predictions.values())
    print(f"Successfully loaded {total_samples} predictions")
    for dataset, preds in predictions.items():
        print(f"  {dataset}: {len(preds)} samples")
    
    return dict(predictions)


def load_ground_truth(dataset_name: str) -> dict:
    """Load ground truth labels
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        dict: {image_name: label_dict}
    """
    label_path = DATASETS_DIR / dataset_name / "label.json"
    
    if not label_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")
    
    print(f"Loading ground truth labels: {label_path}")
    
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"Loaded {len(labels)} ground truth labels")
    return labels


def evaluate_dataset(predictions: dict, ground_truth: dict, dataset_name: str) -> dict:
    """Evaluate a single dataset
    
    Args:
        predictions: {image_name: prediction_dict}
        ground_truth: {image_name: label_dict}
        dataset_name: Dataset name
    
    Returns:
        Evaluation result dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Normalize predictions and ground truth
    normalized_preds = kie_evaluator.normalize_values_of_nested_dict(predictions, normalize_func)
    normalized_gts = kie_evaluator.normalize_values_of_nested_dict(ground_truth, normalize_func)
    
    # Calculate F1 score
    f1_score, class_f1_info, f1_error_info = kie_evaluator.cal_f1_all(normalized_preds, normalized_gts)
    
    # Statistics
    total_pred = len(normalized_preds)
    total_gt = len(normalized_gts)
    matched = len(set(normalized_preds.keys()) & set(normalized_gts.keys()))
    
    print(f"\nDataset statistics:")
    print(f"  Prediction samples: {total_pred}")
    print(f"  Ground truth samples: {total_gt}")
    print(f"  Matched samples: {matched}")
    
    print(f"\nEvaluation results:")
    print(f"  F1 Score: {f1_score:.4f}")
    
    # Return evaluation results
    eval_result = {
        "dataset": dataset_name,
        "summary": {
            "f1_score": f1_score,
            "total_predictions": total_pred,
            "total_ground_truth": total_gt,
            "matched_samples": matched
        },
        "class_f1_score": class_f1_info,
        "f1_error_info": f1_error_info
    }
    
    return eval_result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Evaluate output results from request_openai.py")
    parser.add_argument("--pred", type=str, required=True,
                        help="Prediction result JSONL file path (output from request_openai.py)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Specify dataset name to evaluate (e.g., SIBR, CELL), if not specified, evaluate all datasets")
    parser.add_argument("--output", type=str, default=None,
                        help="Evaluation result output JSON file path (optional)")
    
    args = parser.parse_args()
    
    # Load predictions
    all_predictions = load_predictions(args.pred)
    
    if not all_predictions:
        print("Error: No predictions loaded")
        return
    
    datasets_to_eval = []
    if args.dataset:
        if args.dataset not in all_predictions:
            print(f"Error: Dataset {args.dataset} not found in predictions")
            print(f"Available datasets: {list(all_predictions.keys())}")
            return
        datasets_to_eval = [args.dataset]
    else:
        datasets_to_eval = list(all_predictions.keys())
    
    all_results = {}
    
    for dataset_name in datasets_to_eval:
        try:
            # Load ground truth
            ground_truth = load_ground_truth(dataset_name)
            
            # Get predictions for this dataset
            predictions = all_predictions[dataset_name]
            
            # Evaluate
            eval_result = evaluate_dataset(predictions, ground_truth, dataset_name)
            all_results[dataset_name] = eval_result
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Error: Failed to evaluate dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary Results")
        print(f"{'='*60}")
        
        total_f1 = sum(r["summary"]["f1_score"] for r in all_results.values())
        total_pred = sum(r["summary"]["total_predictions"] for r in all_results.values())
        total_gt = sum(r["summary"]["total_ground_truth"] for r in all_results.values())
        total_matched = sum(r["summary"]["matched_samples"] for r in all_results.values())
        
        avg_f1 = total_f1 / len(all_results)
        
        print(f"\nOverall statistics:")
        print(f"  Number of datasets: {len(all_results)}")
        print(f"  Total prediction samples: {total_pred}")
        print(f"  Total ground truth samples: {total_gt}")
        print(f"  Total matched samples: {total_matched}")
        print(f"\nAverage evaluation results:")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        
        all_results["_summary"] = {
            "num_datasets": len(all_results),
            "average_f1_score": avg_f1,
            "total_predictions": total_pred,
            "total_ground_truth": total_gt,
            "total_matched_samples": total_matched
        }
    
    if args.output:
        output_path = Path(args.output)
    else:
        pred_file = Path(args.pred)
        eval_filename = f"{pred_file.stem}_eval.json"
        output_path = pred_file.parent / eval_filename
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nEvaluation results saved to: {output_path}")

if __name__ == "__main__":
    main()
