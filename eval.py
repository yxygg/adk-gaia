# eval.py
import json
import os
import argparse
import logging
import re
import string
import warnings
from typing import Dict, Any, List

import numpy as np
# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 从 config.py 或直接定义 GAIA 数据目录 ---
try:
    from src.core.config import get_gaia_data_dir
    GAIA_BASE_DIR = get_gaia_data_dir()
except ImportError:
    logger.warning("Could not import config. Using default GAIA data directory './GAIA/2023'")
    GAIA_BASE_DIR = "./GAIA/2023" # 默认路径

GAIA_SPLIT = "validation" # 评估验证集
GAIA_SPLIT_DIR = os.path.join(GAIA_BASE_DIR, GAIA_SPLIT) if GAIA_BASE_DIR else None

# --- 评分函数 (直接从你提供的代码复制) ---
def normalize_number_str(number_str: str) -> float:
    """Converts a string to a float after removing common units/commas."""
    if number_str is None: return float("inf") # 处理 None 输入
    # 确保 number_str 是字符串类型
    number_str = str(number_str)
    # 移除 $, %, ,
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    # 尝试转换为 float
    try:
        # 额外处理带单位的，例如 '17 yards' (只取数字部分)
        match = re.match(r"([-+]?\d*\.?\d+)", number_str)
        if match:
            return float(match.group(1))
        else:
            # 如果没有匹配到数字，尝试直接转换，失败则返回 inf
            return float(number_str)
    except ValueError:
        logger.warning(f"String '{number_str}' cannot be normalized to number str.")
        return float("inf")
    except Exception as e: # 捕获其他可能的错误
         logger.error(f"Unexpected error normalizing number string '{number_str}': {e}")
         return float("inf")


def split_string(
    s: str,
    char_list: list[str] = [",", ";"],
) -> list[str]:
    """Splits a string by any character in char_list."""
    if s is None: return [] # 处理 None 输入
    s = str(s) # 确保是字符串
    pattern = f"[{''.join(re.escape(c) for c in char_list)}]" # 转义特殊字符
    # 分割后去除每个元素两端的空格
    return [item.strip() for item in re.split(pattern, s)]


def normalize_str(input_str: str, remove_punct: bool = True) -> str:
    """Normalizes a string by removing whitespace, punctuation (optional), and lowercasing."""
    if input_str is None: return "" # 处理 None 输入
    input_str = str(input_str) # 确保是字符串
    # Remove all white spaces. Required e.g for seagull vs. sea gull
    no_spaces = re.sub(r"\s+", "", input_str) # 使用 \s+ 匹配一个或多个空白

    # Remove punctuation, if specified.
    if remove_punct:
        # 修正：确保移除所有 Unicode 标点
        # 使用更广泛的标点定义或明确列出要移除的标点
        # Python's string.punctuation 只包含 ASCII 标点
        # 为了安全，我们可以在这里移除所有非字母数字字符（如果适用）
        # 或者坚持使用 string.punctuation 并接受其局限性
        translator = str.maketrans("", "", string.punctuation)
        normalized = no_spaces.lower().translate(translator)
    else:
        normalized = no_spaces.lower()

    # 额外的 normalization: 替换常见单词形式
    # 例如： and -> & (如果需要)
    # 例如： first -> 1st (如果需要)
    # normalized = normalized.replace('and', '&') # 示例
    return normalized


def question_scorer(model_answer: str, ground_truth: str) -> bool:
    """Scores the model answer against the ground truth based on GAIA rules."""
    def is_float(element: any) -> bool:
        try:
            float(str(element)) # 转为字符串再尝试
            return True
        except (ValueError, TypeError):
            return False

    if model_answer is None:
        logger.info(f"Evaluating Model Answer: None | Ground Truth: {ground_truth} -> False")
        return False
    if ground_truth is None:
        logger.warning(f"Ground truth is None for model answer '{model_answer}'. Returning False.")
        return False

    # 确保都是字符串以便后续处理
    model_answer = str(model_answer)
    ground_truth = str(ground_truth)

    # if gt is a number
    if is_float(ground_truth):
        logger.info(f"Evaluating '{model_answer}' as a number (GT: {ground_truth}).")
        # 先对模型答案进行基本清理，移除首尾空格
        model_answer_cleaned = model_answer.strip()
        # 标准化模型答案和标准答案为浮点数
        normalized_ma = normalize_number_str(model_answer_cleaned)
        normalized_gt = float(ground_truth) # GT 已经是数字，直接转换

        # 比较浮点数时考虑精度问题
        is_correct = np.isclose(normalized_ma, normalized_gt)
        logger.info(f"Normalized MA: {normalized_ma}, Normalized GT: {normalized_gt} -> Correct: {is_correct}")
        return is_correct

    # if gt is a list (contains comma or semicolon)
    elif any(char in ground_truth for char in [",", ";"]):
        logger.info(f"Evaluating '{model_answer}' as a comma/semicolon separated list (GT: {ground_truth}).")
        gt_elems = split_string(ground_truth)
        ma_elems = split_string(model_answer)

        logger.info(f"GT elements: {gt_elems} | MA elements: {ma_elems}")

        # Check length is the same
        if len(gt_elems) != len(ma_elems):
            logger.warning(f"Answer lists have different lengths ({len(ma_elems)} vs {len(gt_elems)}), returning False.")
            return False

        # Compare each element
        comparisons = []
        for i, (ma_elem, gt_elem) in enumerate(zip(ma_elems, gt_elems)):
            item_correct = False
            if is_float(gt_elem):
                normalized_ma_elem = normalize_number_str(ma_elem)
                normalized_gt_elem = float(gt_elem)
                item_correct = np.isclose(normalized_ma_elem, normalized_gt_elem)
                logger.info(f"  Item {i+1} (Number): MA='{ma_elem}'({normalized_ma_elem}), GT='{gt_elem}'({normalized_gt_elem}) -> {item_correct}")
            else:
                # Normalize strings *without* removing punctuation for list comparison
                normalized_ma_elem = normalize_str(ma_elem, remove_punct=False)
                normalized_gt_elem = normalize_str(gt_elem, remove_punct=False)
                item_correct = (normalized_ma_elem == normalized_gt_elem)
                logger.info(f"  Item {i+1} (String): MA='{ma_elem}'({normalized_ma_elem}), GT='{gt_elem}'({normalized_gt_elem}) -> {item_correct}")
            comparisons.append(item_correct)

        all_correct = all(comparisons)
        logger.info(f"Overall list comparison result: {all_correct}")
        return all_correct

    # if gt is a string
    else:
        logger.info(f"Evaluating '{model_answer}' as a string (GT: {ground_truth}).")
        # Normalize strings *with* removing punctuation for simple string comparison
        normalized_ma = normalize_str(model_answer, remove_punct=True)
        normalized_gt = normalize_str(ground_truth, remove_punct=True)
        is_correct = (normalized_ma == normalized_gt)
        logger.info(f"Normalized MA: '{normalized_ma}', Normalized GT: '{normalized_gt}' -> Correct: {is_correct}")
        return is_correct


# --- Helper Functions ---
def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads data from a JSON Lines file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in {file_path}: {line.strip()}")
        logger.info(f"Successfully loaded {len(data)} records from {file_path}")
        return data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return []
    except Exception as e:
        logger.exception(f"Error reading JSONL file {file_path}: {e}")
        return []

# --- Main Evaluation Logic ---
def evaluate_results(results_file: str, metadata_file: str):
    """Evaluates the results against the ground truth."""
    logger.info(f"Starting evaluation...")
    logger.info(f"Results file: {results_file}")
    logger.info(f"Metadata file: {metadata_file}")

    results_data = load_jsonl(results_file)
    metadata_data = load_jsonl(metadata_file)

    if not results_data or not metadata_data:
        logger.error("Could not load results or metadata. Aborting evaluation.")
        return

    # Create a dictionary for quick lookup of ground truth by task_id
    ground_truths = {task['task_id']: task.get('Final Answer') or task.get('Final answer')
                     for task in metadata_data}

    correct_count = 0
    total_count = 0
    missing_gt_count = 0
    evaluation_details = [] # To store details for each task

    # Iterate through the model results
    for result in results_data:
        task_id = result.get("task_id")
        model_answer = result.get("model_answer")

        if not task_id:
            logger.warning(f"Skipping result with missing task_id: {result}")
            continue

        total_count += 1
        ground_truth = ground_truths.get(task_id)

        if ground_truth is None:
            logger.warning(f"No ground truth found for task_id: {task_id}. Skipping evaluation for this task.")
            missing_gt_count += 1
            is_correct = None # Mark as unevaluated
        else:
            logger.info(f"\n--- Evaluating Task ID: {task_id} ---")
            is_correct = question_scorer(model_answer, ground_truth)
            if is_correct:
                correct_count += 1
            logger.info(f"Result for Task ID {task_id}: {'Correct' if is_correct else 'Incorrect'}")
            logger.info(f"----------------------------------")


        evaluation_details.append({
            "task_id": task_id,
            "model_answer": model_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "error_in_result": result.get("error") # Include error from result file
        })


    # --- Print Summary ---
    logger.info("\n--- Evaluation Summary ---")
    logger.info(f"Total results processed: {total_count}")
    evaluatable_count = total_count - missing_gt_count
    logger.info(f"Results with missing ground truth: {missing_gt_count}")
    logger.info(f"Results evaluated: {evaluatable_count}")
    logger.info(f"Correct answers: {correct_count}")

    if evaluatable_count > 0:
        accuracy = (correct_count / evaluatable_count) * 100
        logger.info(f"Accuracy: {accuracy:.2f}%")
    else:
        logger.info("Accuracy: N/A (no results could be evaluated)")

    logger.info("------------------------")

    # 可选：将详细评估结果保存到文件
    # eval_output_file = results_file.replace(".jsonl", "_evaluation.jsonl")
    # try:
    #     with open(eval_output_file, 'w', encoding='utf-8') as f:
    #         for detail in evaluation_details:
    #             f.write(json.dumps(detail) + '\n')
    #     logger.info(f"Detailed evaluation results saved to: {eval_output_file}")
    # except IOError as e:
    #     logger.error(f"Could not write detailed evaluation results to {eval_output_file}: {e}")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GAIA agent results.")
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the JSON Lines file containing the agent's results (e.g., gaia_validation_results_*.jsonl)."
    )
    parser.add_argument(
        "--metadata_file",
        type=str,
        default=os.path.join(GAIA_SPLIT_DIR, "metadata.jsonl") if GAIA_SPLIT_DIR else None,
        help=f"Path to the GAIA metadata file containing ground truths (default: attempts to find it in {GAIA_SPLIT_DIR})."
    )

    args = parser.parse_args()

    if not args.metadata_file:
         logger.critical(f"Metadata file path could not be determined. Please specify using --metadata_file or ensure GAIA_BASE_DIR is correct.")
    elif not os.path.exists(args.results_file):
        logger.critical(f"Results file not found: {args.results_file}")
    elif not os.path.exists(args.metadata_file):
         logger.critical(f"Metadata file not found: {args.metadata_file}")
    else:
        evaluate_results(args.results_file, args.metadata_file)