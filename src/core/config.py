# src/core/config.py
import json
import os
from typing import Dict, Any, Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json') # 指向根目录的 config.json

def load_config() -> Dict[str, Any]:
    """Loads the configuration from config.json."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        return config_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {CONFIG_PATH}: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while loading config: {e}")

# 加载一次配置供全局使用
try:
    APP_CONFIG = load_config()
except (FileNotFoundError, ValueError, RuntimeError) as e:
    print(f"Error loading application configuration: {e}")
    # 提供一个默认的最小配置，以防文件加载失败，但最好是让程序失败并提示用户修复
    APP_CONFIG = {
        "orchestrator_model": "gemini-2.5-pro-preview-03-25",
        "specialist_model_flash": "gemini-2.5-flash-preview-04-17",
        "specialist_model_pro": "gemini-2.5-pro-preview-03-25",
        "ollama_model": None,
        "gaia_data_dir": "./GAIA/2023"
    }
    print("Warning: Using default fallback configuration.")


def get_model(model_key: str) -> Optional[str]:
    """Safely retrieves a model name from the loaded configuration."""
    return APP_CONFIG.get(model_key)

def get_gaia_data_dir() -> Optional[str]:
    """Retrieves the GAIA data directory path."""
    return APP_CONFIG.get("gaia_data_dir")

def get_api_port() -> int:
    """Retrieves the API port number from the configuration."""
    # 提供一个默认值，以防配置中缺少该键
    return APP_CONFIG.get("api_port", 8000)

def get_runner_strategy() -> str:
    """Retrieves the runner strategy, defaulting to 'all'."""
    strategy = APP_CONFIG.get("runner_strategy", "all")
    if strategy not in ["all", "single", "first_n"]:
        print(f"Warning: Invalid runner_strategy '{strategy}' in config. Defaulting to 'all'.")
        return "all"
    return strategy

def get_runner_task_id() -> Optional[str]:
    """Retrieves the specific task ID for the 'single' strategy."""
    return APP_CONFIG.get("runner_task_id")

def get_runner_first_n() -> Optional[int]:
    """Retrieves the number of tasks for the 'first_n' strategy."""
    n = APP_CONFIG.get("runner_first_n")
    if n is not None:
        try:
            return int(n)
        except (ValueError, TypeError):
            print(f"Warning: Invalid runner_first_n value '{n}' in config. Ignoring.")
            return None
    return None