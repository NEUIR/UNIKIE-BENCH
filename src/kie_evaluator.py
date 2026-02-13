import json
import re
from typing import Any, Dict, List, Tuple, Union

from collections import Counter


def flatten(data: dict):
    """
    Convert Dictionary into Non-nested Dictionary
    Example:
        input(dict)
            {
                "menu": [
                    {"name" : ["cake"], "count" : ["2"]},
                    {"name" : ["juice"], "count" : ["1"]},
                ]
            }
        output(list)
            [
                ("menu.name", "cake"),
                ("menu.count", "2"),
                ("menu.name", "juice"),
                ("menu.count", "1"),
            ]
    """
    flatten_data = list()

    def _flatten(value, key=""):
        if type(value) is dict:
            for child_key, child_value in value.items():
                _flatten(child_value, f"{key}.{child_key}" if key else child_key)
        elif type(value) is list:
            for value_item in value:
                _flatten(value_item, key)
        else:
            flatten_data.append((key, value))

    _flatten(data)
    return flatten_data


def normalize_dict(data: Union[Dict, List, Any]):
    """
    Sort by value, while iterate over element if data is list
    """
    # if not data:
    #     return {}

    if isinstance(data, dict):
        new_data = dict()
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value

    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            new_data = []
            for item in data:
                item = normalize_dict(item)
                if item:
                    new_data.append(item)
        else:
            new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    else:
        new_data = [str(data).strip()]
    return new_data


def cal_f1_all(preds, answers):
    """
    Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives,
    false negatives and false positives
    """
    metric_info, error_info = {}, {}
    total_tp, total_fn_or_fp = 0, 0
    for file_name, answer in answers.items():
        sample_error_info = {"fp": [], "fn": [], "tp": []}
        pred = preds.get(file_name, {})
        pred, answer = flatten(normalize_dict(pred)), flatten(normalize_dict(answer))
        for field in pred:
            field_name = field[0]
            if field_name not in metric_info:
                metric_info[field_name] = {"total_tp": 0, "total_fn_or_fp": 0}
            if field in answer:
                total_tp += 1
                metric_info[field_name]["total_tp"] += 1
                sample_error_info["tp"].append(field)
                answer.remove(field)
            else:
                total_fn_or_fp += 1
                metric_info[field_name]["total_fn_or_fp"] += 1
                sample_error_info["fp"].append(field)

        total_fn_or_fp += len(answer)
        for field in answer:
            field_name = field[0]
            if field_name not in metric_info:
                metric_info[field_name] = {"total_tp": 0, "total_fn_or_fp": 0}
            metric_info[field_name]["total_fn_or_fp"] += 1
            sample_error_info["fn"].append(field)

        sample_error_num = sum([len(v) for k, v in sample_error_info.items() if k != "tp"])
        if sample_error_num > 0:
            sample_error_info["error_num"] = sample_error_num
            error_class_list = ["counter_" + x[0] for x in (sample_error_info["fn"] + sample_error_info["fp"])]
            counter = Counter(error_class_list)
            sample_error_info["error_info"] = dict(counter)
            error_info[file_name] = sample_error_info

    # summary
    for field_name, field_info in metric_info.items():
        field_tp, field_fn_or_fp = field_info["total_tp"], field_info["total_fn_or_fp"]
        metric_info[field_name]["acc"] = field_tp / (field_tp + field_fn_or_fp / 2 + 1e-6)

    print("donut_evaluator: total_tp: {}, total_fn_or_fp: {}, ptd_num: {}, gt_num: {}".format(total_tp, total_fn_or_fp,
                                                                                              len(preds), len(answers)))
    error_info = {k: v for k, v in
                  sorted(error_info.items(), key=lambda item: item[1].get("error_num", 0), reverse=True)}
    metric_info = {k: v for k, v in
                   sorted(metric_info.items(), key=lambda item: item[1].get("total_fn_or_fp", 0), reverse=True)}
    return total_tp / (total_tp + total_fn_or_fp / 2 + 1e-6), metric_info, error_info



def normalize_values_of_nested_dict(d, normalize_func):
    """
    """
    if isinstance(d, dict):
        return {k: normalize_values_of_nested_dict(v, normalize_func) for k, v in d.items()}
    elif isinstance(d, list):
        # Modified: Added handling for string elements in lists
        return [normalize_values_of_nested_dict(x, normalize_func) if isinstance(x, dict) else normalize_func(x) if isinstance(x, str) else x for x in d]
    elif isinstance(d, str):
        return normalize_func(d)
    else:
        return d

def eval_donut(pdt_info, gt_info, normalize_func=None, data_name=None):
    """
    """
    if normalize_func is not None:
        print("--> info: normalize_func executed.")
        pdt_info = normalize_values_of_nested_dict(pdt_info, normalize_func)
        gt_info = normalize_values_of_nested_dict(gt_info, normalize_func)

    f1_score, class_eval_info, error_info = cal_f1_all(pdt_info, gt_info)
    eval_info = {"f1_score": f1_score, "class_f1_score": class_eval_info,
                 "f1_error_info": error_info}
    print(data_name, "f1_score", f1_score)
    return eval_info


def post_process_to_json(qwen_info_str, file_name=None):
    try:
        if "```json" in qwen_info_str:
            if "```" not in qwen_info_str:
                qwen_info_str += "```"
            qwen_info_group = re.search(r'```json(.*?)```', qwen_info_str, re.DOTALL)
            json_str = qwen_info_group.group(1).strip().replace("\n", "")
        else:
            json_str = qwen_info_str.strip().replace("\n", "")
        json_data = json.loads(json_str)
        return json_data
    except Exception as err:  # noqa: F841
        return None


def fullwidth_to_halfwidth(text):
    result = ''
    for char in text:
        code_point = ord(char)
        if code_point == 0x3000:
            code_point = 0x0020
        elif code_point == 0xFFE5:
            code_point = 0x00A5
        elif code_point == 0x2014:
            code_point = 0x002D
        elif code_point == 0x2103:
            result += chr(0x00B0) + 'C'
            continue
        elif 0xFF01 <= code_point <= 0xFF5E:
            code_point -= 0xFEE0
        result += chr(code_point)
    result = result.replace("、", ",")
    result = result.replace("-", "")
    result = result.replace("–","")
    result = result.replace("’","'")
    result = result.rstrip("。.")
    return result


def remove_unnecessary_spaces(text):
    if "```json" in text:
        json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()
    elif "```" in text:
        code_match = re.search(r'```\s*(.*?)\s*```', text, re.DOTALL)
        if code_match:
            text = code_match.group(1).strip()
    text = re.sub(r'\s+', '', text)
    return text


if __name__ == '__main__':
    pass