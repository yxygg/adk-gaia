# src/agents/code_executor.py
from google.adk.agents import LlmAgent
from google.adk.tools import built_in_code_execution
from src.core.config import get_model

CODE_EXECUTOR_MODEL = get_model("specialist_model_flash") # Flash 模型通常足够

if not CODE_EXECUTOR_MODEL:
    raise ValueError("Model for Code Executor Agent not found in configuration.")

code_executor_agent = LlmAgent(
    name="CodeExecutorAgent",
    model=CODE_EXECUTOR_MODEL,
    description=(
        "Specializes in executing Python code snippets provided to it. "
        "Useful for calculations, data manipulation, or running algorithms."
    ),
    instruction=(
        "You are a secure code execution environment. Your ONLY capability is to run Python code using the "
        "`built_in_code_execution` tool when explicitly requested or when a calculation is needed. "
        "Receive Python code, execute it, and return the standard output or error. "
        "Do NOT attempt to search the web, access files, or perform any other actions. "
        "Ensure the code provided for execution is complete and runnable. "
        "If asked to perform a calculation, generate the necessary Python code and execute it."
        "Return ONLY the result of the code execution (stdout) or the error message."
    ),
    tools=[built_in_code_execution] # 集成 ADK 内置的代码执行工具
)

print(f"CodeExecutorAgent initialized with model: {CODE_EXECUTOR_MODEL}")