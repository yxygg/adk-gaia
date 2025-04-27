# run_gaia.py
import json
import os
import time
import requests
import logging
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- 从配置模块导入 ---
try:
    from src.core.config import (
        get_gaia_data_dir,
        get_api_port,
        get_runner_strategy,
        get_runner_task_id,
        get_runner_first_n
    )
except ImportError as e:
    print(f"Error importing configuration: {e}. Make sure src/core/config.py exists and is correct.")
    # Provide dummy getters if import fails to allow basic script structure check
    get_gaia_data_dir = lambda: "./GAIA/2023"
    get_api_port = lambda: 8000
    get_runner_strategy = lambda: "all"
    get_runner_task_id = lambda: None
    get_runner_first_n = lambda: None


# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- 常量和配置加载 ---
API_HOST = os.getenv("API_HOST", "http://localhost") # Allow overriding host via env var
API_PORT = get_api_port()
API_BASE_URL = f"{API_HOST}:{API_PORT}"
CHAT_ENDPOINT = f"{API_BASE_URL}/chat"
GAIA_SPLIT = "validation" # 或 "test"
OUTPUT_FILE = f"gaia_{GAIA_SPLIT}_results_{int(time.time())}.jsonl" # 添加时间戳避免覆盖
MAX_WORKERS = 1
USER_ID_PREFIX = "gaia_runner"

# 运行策略配置
RUNNER_STRATEGY = get_runner_strategy()
RUNNER_TASK_ID = get_runner_task_id()
RUNNER_FIRST_N = get_runner_first_n()

GAIA_BASE_DIR = get_gaia_data_dir()
GAIA_SPLIT_DIR = os.path.join(GAIA_BASE_DIR, GAIA_SPLIT) if GAIA_BASE_DIR else None

# --- 加载 GAIA 数据 ---
def load_gaia_data(metadata_file: str) -> List[Dict[str, Any]]:
    """Loads GAIA tasks from the specified metadata file."""
    if not GAIA_SPLIT_DIR or not os.path.exists(metadata_file):
        logger.error(f"GAIA metadata file not found at {metadata_file} or directory not configured.")
        return []
    tasks = []
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    tasks.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line {i+1} in {metadata_file}: {line.strip()}")
        logger.info(f"Loaded {len(tasks)} tasks from {metadata_file}")
        return tasks
    except Exception as e:
        logger.exception(f"Error reading GAIA metadata file {metadata_file}: {e}")
        return []

# --- 调用 Agent API ---
def call_agent_api(task: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Calls the FastAPI endpoint for a single GAIA task."""
    task_id = task["task_id"]
    logger.info(f"Submitting task {task_id}...")
    request_payload = {
        "user_id": user_id,
        "task_id": task_id,
        "question": task["Question"],
        "file_name": task.get("file_name"),
        # 为每个任务创建独立会话可能更清晰，避免状态污染
        "session_id": f"session_{task_id}_{uuid.uuid4()}"
    }
    result = {
        "task_id": task_id,
        "model_answer": None,
        "reasoning_trace_summary": None, # 存储简化追踪或错误信息
        "error": None,
        "api_response_status": None,
        "ground_truth": task.get("Final Answer")
    }
    start_time = time.time()
    try:
        response = requests.post(CHAT_ENDPOINT, json=request_payload, timeout=600) # 超时增加到 10 分钟
        result["api_response_status"] = response.status_code
        response.raise_for_status()

        response_data = response.json()
        result["model_answer"] = response_data.get("model_answer")
        result["error"] = response_data.get("error") # API 返回的错误

        # 存储部分追踪信息，例如错误或最后几个事件
        trace = response_data.get("reasoning_trace")
        if result["error"] and trace:
             # 只记录包含错误的追踪部分
             result["reasoning_trace_summary"] = [evt for evt in trace if evt.get("error")] or trace[-1:] # 最后一条
        # elif trace:
        #     # 可以选择记录最后几条或特定类型的事件
        #     result["reasoning_trace_summary"] = trace[-3:] # 例如最后3条

    except requests.exceptions.Timeout:
        logger.error(f"API request timed out for task {task_id}")
        result["error"] = "API request timed out"
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for task {task_id}: {e}")
        result["error"] = f"API request failed: {str(e)}"
    except json.JSONDecodeError as e:
        # 检查响应状态码和内容
        response_text = response.text if 'response' in locals() else "N/A"
        logger.error(f"Failed to decode API JSON response for task {task_id}. Status: {result['api_response_status']}, Response text: {response_text[:200]}... Error: {e}")
        result["error"] = f"Failed to decode API response (status {result['api_response_status']})"
    except Exception as e:
        logger.exception(f"Unexpected error processing task {task_id}: {e}") # 使用 logger.exception 包含 traceback
        result["error"] = f"Unexpected error: {str(e)}"
    finally:
        duration = time.time() - start_time
        status_msg = "Success" if result["model_answer"] is not None and result["error"] is None else "Failed"
        logger.info(f"Task {task_id} finished in {duration:.2f}s. Status: {status_msg}. Answer: {result['model_answer']}. Error: {result['error']}")

    return result

# --- 主函数 ---
def main():
    """主执行函数，加载数据，根据策略运行任务，并实时写入结果。"""
    logger.info("--- Starting GAIA Agent Runner ---")

    if not GAIA_SPLIT_DIR:
        logger.critical("GAIA data directory path could not be determined. Exiting.")
        return

    metadata_file = os.path.join(GAIA_SPLIT_DIR, "metadata.jsonl")
    all_tasks = load_gaia_data(metadata_file)

    if not all_tasks:
        logger.critical("No tasks loaded. Exiting.")
        return

    # --- 应用运行策略 ---
    tasks_to_run = []
    logger.info(f"Applying runner strategy: {RUNNER_STRATEGY}")
    if RUNNER_STRATEGY == "single":
        if not RUNNER_TASK_ID:
            logger.critical("Runner strategy is 'single' but 'runner_task_id' is not set in config.json. Exiting.")
            return
        tasks_to_run = [task for task in all_tasks if task.get("task_id") == RUNNER_TASK_ID]
        if not tasks_to_run:
            logger.error(f"Task ID '{RUNNER_TASK_ID}' not found in {metadata_file}. Exiting.")
            return
        logger.info(f"Selected single task: {RUNNER_TASK_ID}")
    elif RUNNER_STRATEGY == "first_n":
        if RUNNER_FIRST_N is None or RUNNER_FIRST_N <= 0:
            logger.critical("Runner strategy is 'first_n' but 'runner_first_n' is not a positive integer in config.json. Exiting.")
            return
        tasks_to_run = all_tasks[:RUNNER_FIRST_N]
        logger.info(f"Selected first {len(tasks_to_run)} tasks (requested: {RUNNER_FIRST_N}).")
    else: # "all" 或默认
        tasks_to_run = all_tasks
        logger.info(f"Selected all {len(tasks_to_run)} tasks.")

    if not tasks_to_run:
         logger.warning("No tasks selected to run based on the current strategy.")
         return

    # --- 检查 API 可达性 ---
    try:
        logger.info(f"Pinging API server at {API_BASE_URL}...")
        response = requests.get(API_BASE_URL, timeout=10) # 增加超时
        if response.status_code == 200:
            logger.info(f"Successfully connected to API at {API_BASE_URL}")
        else:
             logger.warning(f"API at {API_BASE_URL} returned status {response.status_code}. Attempting to proceed...")
    except requests.exceptions.ConnectionError:
        logger.critical(f"Error: Could not connect to the agent API at {API_BASE_URL}.")
        logger.critical(f"Please ensure the FastAPI server is running on port {API_PORT}.")
        return
    except requests.exceptions.Timeout:
         logger.critical(f"Error: Connection to the agent API at {API_BASE_URL} timed out.")
         return
    except Exception as e:
         logger.error(f"An error occurred while checking API status: {e}")
         logger.warning("Proceeding without API status confirmation.")


    # --- 执行任务并实时写入结果 ---
    processed_count = 0
    success_count = 0
    error_count = 0
    logger.info(f"Starting processing of {len(tasks_to_run)} tasks. Results will be written to {OUTPUT_FILE}")

    # 使用 'a' 模式打开文件以追加写入
    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as outfile, ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(call_agent_api, task, f"{USER_ID_PREFIX}_{task['task_id']}"): task['task_id'] for task in tasks_to_run}

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    # 实时写入结果
                    outfile.write(json.dumps(result) + '\n')
                    outfile.flush() # 确保立即写入磁盘

                    processed_count += 1
                    if result.get("model_answer") is not None and result.get("error") is None:
                        success_count += 1
                    else:
                        error_count += 1

                except Exception as e:
                    logger.exception(f"Error retrieving result future for task {task_id}: {e}")
                    error_result = {
                        "task_id": task_id,
                        "model_answer": None,
                        "reasoning_trace_summary": f"Future execution failed: {str(e)}",
                        "error": f"Future execution failed: {str(e)}",
                        "api_response_status": None,
                        "ground_truth": next((t.get("Final Answer") for t in tasks_to_run if t["task_id"] == task_id), None)
                    }
                    outfile.write(json.dumps(error_result) + '\n')
                    outfile.flush()
                    processed_count += 1
                    error_count += 1

            logger.info("All submitted tasks futures completed.")

    except IOError as e:
        logger.critical(f"Error writing results to {OUTPUT_FILE}: {e}")
    except Exception as e:
        logger.critical(f"An unexpected error occurred during task execution: {e}", exc_info=True)


    # --- 最终总结 ---
    logger.info("--- Final Summary ---")
    logger.info(f"Runner strategy: {RUNNER_STRATEGY}")
    if RUNNER_STRATEGY == "single": logger.info(f"Target task ID: {RUNNER_TASK_ID}")
    if RUNNER_STRATEGY == "first_n": logger.info(f"Target N: {RUNNER_FIRST_N}")
    logger.info(f"Total tasks selected: {len(tasks_to_run)}")
    logger.info(f"Total tasks processed: {processed_count}")
    logger.info(f"Successful answers: {success_count}")
    logger.info(f"Failed/Errored tasks: {error_count}")
    logger.info(f"Results saved to: {OUTPUT_FILE}")
    logger.info("---------------------")

if __name__ == "__main__":
    import uuid # 确保导入 uuid
    main()