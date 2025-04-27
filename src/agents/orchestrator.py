# src/agents/orchestrator.py
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from src.core.config import get_model, get_gaia_data_dir
import os
import logging # 添加日志

# 导入需要被包装的 Agent 实例
from .web_researcher import web_researcher_agent
from .code_executor import code_executor_agent
from .file_processor import file_processor_agent
# from .calculator import calculator_agent

logger = logging.getLogger(__name__) # 添加日志记录器

ORCHESTRATOR_MODEL = get_model("orchestrator_model")
# GAIA_DATA_DIR = get_gaia_data_dir() # 不再需要在 Orchestrator 中直接使用

if not ORCHESTRATOR_MODEL:
    raise ValueError("Model for Orchestrator Agent not found in configuration.")

# 将 Specialist Agents 包装成 AgentTool
web_researcher_tool = agent_tool.AgentTool(agent=web_researcher_agent)
code_executor_tool = agent_tool.AgentTool(agent=code_executor_agent)
file_processor_tool = agent_tool.AgentTool(agent=file_processor_agent)
# calculator_tool = agent_tool.AgentTool(agent=calculator_agent)

# 定义 Orchestrator Agent
orchestrator_agent = LlmAgent(
    name="GAIAOrchestratorAgent",
    model=ORCHESTRATOR_MODEL,
    description=(
        "The main coordinator agent for solving GAIA benchmark tasks. "
        "It understands the task, plans the steps, potentially checks for files in session state, " # 强调检查状态
        "and delegates sub-tasks to specialized agents."
    ),
    instruction=( # *** 简化文件处理委托指令 ***
        "You are a master agent designed to solve complex questions from the GAIA benchmark. "
        "Your goal is to accurately answer the user's question.\n"
        "Follow these steps:\n"
        "1.  **Understand the Task:** Carefully analyze the user's question. Check if a relevant file path exists in `session.state['gaia_file_path']`. Note it down if present.\n"
        "2.  **Plan Execution:** Break down the problem into logical steps. Determine if you need to search the web, execute code, process the file (if one exists), or perform calculations.\n"
        "3.  **Delegate Tasks:** Use the available tools (`WebResearcherAgent`, `CodeExecutorAgent`, `FileProcessorAgent`, etc.) to execute each step.\n"
        "    - Use `WebResearcherAgent` for web searches.\n"
        "    - Use `CodeExecutorAgent` for running Python code.\n"
        "    - **If a file needs processing (based on step 1):** Call `FileProcessorAgent`. Provide a clear instruction in the `request` argument describing exactly *what* needs to be done with the file (e.g., 'Find the oldest Blu-Ray title', 'Summarize the document', 'Transcribe the audio'). **Do NOT pass the file path itself as an argument;** the `FileProcessorAgent` knows to get it from the session state.\n"
        "    - (Future) Use `CalculatorAgent` for calculations.\n"
        "4.  **Manage Data:** Use the results returned by the tools. Store intermediate results in `session.state` only if absolutely necessary for a subsequent step.\n"
        "5.  **Synthesize and Format:** Combine information from tool results to formulate the final answer. Ensure the answer is concise, factual, and strictly adheres to the GAIA format (number, string, comma-separated list) without extra explanations.\n"
        "6.  **Final Output:** Present your final answer using the template: 'FINAL ANSWER: [Your Final Answer]'."
    ),
    tools=[
        web_researcher_tool,
        code_executor_tool,
        file_processor_tool,
        # calculator_tool,
    ],
)

logger.info(f"OrchestratorAgent initialized with model: {ORCHESTRATOR_MODEL}")
logger.info(f"Orchestrator Tools: {[tool.name for tool in orchestrator_agent.tools]}")