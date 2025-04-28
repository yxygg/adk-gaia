# src/agents/orchestrator.py
from google.adk.agents import LlmAgent
from google.adk.tools import agent_tool
from src.core.config import get_model

import logging

# 导入需要被包装的 Agent 实例
from .web_researcher import web_researcher_agent
from .code_executor import code_executor_agent
from .file_processor import file_processor_agent
# from .calculator import calculator_agent

logger = logging.getLogger(__name__)

ORCHESTRATOR_MODEL = get_model("orchestrator_model")

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
        "It understands the task, plans the steps, extracts file paths from the request, " # 新增提取路径
        "and delegates sub-tasks to specialized agents."
    ),
    instruction=(
        "You are a master agent designed to solve complex questions, including those from the GAIA benchmark. "
        "Your goal is to accurately answer the user's question.\n"
        "Follow these steps:\n"
        "1.  **Understand the Task & Identify Files:** Carefully analyze the user's question. Look for mentions of file paths, especially absolute paths possibly indicated by '[System Note: ...]' or similar phrasing. Note down the **exact absolute path** if found.\n"
        "2.  **Plan Execution:** Break down the problem into logical steps. Determine if you need to search the web, execute code, process the identified file(s), or perform calculations.\n"
        "3.  **Delegate Tasks:** Use the available tools (`WebResearcherAgent`, `CodeExecutorAgent`, `FileProcessorAgent`) to execute each step.\n"
        "    - Use `WebResearcherAgent` for web searches (provide the search query as the `query` argument).\n"
        "    - Use `CodeExecutorAgent` for running Python code (provide the code snippet as the `code` argument).\n"
        "    - **If a file needs processing (based on step 1):** Call the `FileProcessorAgent` tool.\n"
        "        - Construct a **single string** for the `request` argument. This string MUST contain:\n"
        "            a. The **exact absolute file path** identified in step 1.\n"
        "            b. A clear description of **what action** needs to be performed on the file (e.g., 'Summarize the document', 'Extract the table containing sales data', 'Find the oldest Blu-Ray title', 'Transcribe the audio').\n"
        "            c. (Optional) If processing a spreadsheet and filtering is needed, include the pandas query string within the request string (e.g., 'Extract rows where Category is Electronics from /path/to/data.xlsx using query \\'Category == \"Electronics\"\\'').\n"
        "            d. (Optional) If processing a ZIP archive and a specific file needs extraction, include the target filename within the request string (e.g., 'Extract content from report.txt inside /path/to/archive.zip using target_filename report.txt').\n"
        "        - **Example `request` string:** 'Please find the total sales amount from the spreadsheet located at /data/financials.xlsx using the query \\'Region == \"North\"\\''\n"
        "        - **Example `request` string:** 'Analyze the sentiment of the document at /docs/review.pdf'\n"
        "        - **Example `request` string:** 'List the contents of the archive /archive/data.zip'\n"
        "        - **Example `request` string:** 'Extract the file main.py from the archive /archive/code.zip using target_filename main.py'\n"
        "    - (Future) Use `CalculatorAgent` for calculations.\n"
        "4.  **Manage Data:** Use the results returned by the tools. Store intermediate results only if necessary.\n"
        "5.  **Synthesize and Format:** Combine information from tool results to formulate the final answer. Ensure the answer is concise, factual, and strictly adheres to the GAIA format (number, string, comma-separated list) without extra explanations, unless asked.\n"
        "6.  **Final Output:** Present your final answer using the template: 'FINAL ANSWER: [Your Final Answer]'."
    ),
    tools=[
        web_researcher_tool,
        code_executor_tool,
        file_processor_tool,
        # calculator_tool,
    ],
    # 考虑增加 max_output_tokens 如果模型有时过早停止
    # generate_content_config=types.GenerateContentConfig(max_output_tokens=8192) # 示例
)

logger.info(f"OrchestratorAgent initialized with model: {ORCHESTRATOR_MODEL}")
logger.info(f"Orchestrator Tools: {[tool.name for tool in orchestrator_agent.tools]}")