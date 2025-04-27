# src/api.py
import asyncio
import uuid
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
import os
import logging
import traceback
# --- 配置日志 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 加载 .env 文件中的环境变量
# 确保在导入任何使用环境变量的模块之前加载
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(dotenv_path=dotenv_path)

# 检查 GOOGLE_API_KEY 是否已设置
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_GOOGLE_API_KEY":
   print("\n--- WARNING ---")
   print("GOOGLE_API_KEY is not set or is still the placeholder in .env file.")
   print("Please obtain an API key from Google AI Studio (https://aistudio.google.com/app/apikey) and update the .env file.")
   print("---------------\n")
   # 考虑在这里退出或抛出更严重的错误，取决于你的策略
   # exit(1)

# --- ADK Imports (修正后的路径) ---
from google.adk.runners import Runner # <--- 正确的导入路径
# run_sync 和 run_async 是 Runner 类的方法或通过 runner 实例调用，而不是直接从模块导入
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.genai.types import Content, Part

# --- 导入我们的 Orchestrator Agent ---
try:
    from src.agents import orchestrator_agent
    from src.core.config import get_gaia_data_dir
except ImportError as e:
    print(f"Error importing agents or config: {e}")
    print("Please ensure the project structure is correct and all __init__.py files exist.")
    # 提供一个假的 agent 以便 FastAPI 启动，但会报错
    from google.adk.agents import LlmAgent
    orchestrator_agent = LlmAgent(name="DummyAgent", model="gemini-2.0-flash", instruction="Error loading real agent.")
    get_gaia_data_dir = lambda: None # Dummy function


# --- API Models ---
class ChatRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user.")
    session_id: Optional[str] = Field(None, description="Identifier for the chat session. If None, a new session is created.")
    task_id: str = Field(..., description="The GAIA task ID.")
    question: str = Field(..., description="The question from the GAIA task.")
    file_name: Optional[str] = Field(None, description="Optional file name associated with the task.")
    # state: Optional[Dict[str, Any]] = Field({}, description="Initial or existing session state.") # 考虑是否需要从外部传入 state

class ChatResponse(BaseModel):
    session_id: str
    model_answer: Optional[str] = None
    reasoning_trace: Optional[List[Dict[str, Any]]] = None # 或者直接用 Any
    error: Optional[str] = None

# --- FastAPI App ---
app = FastAPI(
    title="GAIA Solver Agent API",
    description="API endpoint to interact with the ADK-based GAIA solving agent system.",
)

# --- ADK Setup ---
# 使用内存会话服务进行演示，生产环境可能需要持久化存储
session_service = InMemorySessionService()
APP_NAME = "gaia_solver_app" # 应用名称

# Runner 实例
runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=session_service,
)

GAIA_BASE_DIR = get_gaia_data_dir()
GAIA_VALIDATION_DIR = os.path.join(GAIA_BASE_DIR, "validation") if GAIA_BASE_DIR else None


# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Receives a GAIA task and returns the agent's final answer."""
    user_id = request.user_id
    task_id = request.task_id
    session_id = request.session_id or str(uuid.uuid4())

    # --- Session Management ---
    try:
        session = session_service.get_session(
            app_name=APP_NAME, user_id=user_id, session_id=session_id
        )
    except KeyError:
        session = None
    except Exception as e:
        logger.warning(f"Error retrieving session {session_id}: {e}")
        session = None

    initial_state = {}

    # --- Prepare and Validate File Path ---
    gaia_file_path = None
    # GAIA_VALIDATION_DIR 来自全局变量，可能是相对路径
    if request.file_name and GAIA_VALIDATION_DIR:
        # 1. 获取 GAIA 验证目录的绝对路径
        abs_gaia_validation_dir = os.path.abspath(GAIA_VALIDATION_DIR)

        if os.path.exists(abs_gaia_validation_dir) and os.path.isdir(abs_gaia_validation_dir):
            # 2. 计算潜在文件的绝对路径
            potential_path = os.path.abspath(os.path.join(abs_gaia_validation_dir, request.file_name))

            # 3. 安全检查：确保潜在路径在预期的验证目录下 (现在比较两个绝对路径)
            #    并且文件确实存在
            if os.path.commonpath([abs_gaia_validation_dir]) == os.path.commonpath([abs_gaia_validation_dir, potential_path]) and os.path.isfile(potential_path):
                 gaia_file_path = potential_path # 存储绝对路径
                 logger.info(f"Validated file path for task {task_id}: {gaia_file_path}")
            else:
                 logger.warning(f"File '{request.file_name}' not found within or is outside expected directory '{abs_gaia_validation_dir}'. Potential path checked: {potential_path}")
                 # 不设置 gaia_file_path，让 Agent 自行处理
        else:
            logger.warning(f"GAIA validation directory does not exist or is not a directory: {abs_gaia_validation_dir}")

    # --- 更新会话状态 ---
    # 总是设置 gaia_file_path 键，即使它是 None，这样 Agent 就知道有没有文件
    initial_state["gaia_file_path"] = gaia_file_path

    if not session:
        logger.info(f"Creating new session: {session_id} for user: {user_id}")
        session = session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
            state=initial_state # 传递包含 gaia_file_path 的初始状态
        )
    else:
        logger.info(f"Using existing session: {session_id}")
        # 更新现有会话的状态（特别是 gaia_file_path 可能变化）
        session.state.update(initial_state)
        # 对于 InMemorySessionService，修改 session.state 会直接生效，无需 update_session

    # --- Prepare Input ---
    user_message = request.question
    content = Content(role='user', parts=[Part(text=user_message)])

    # --- Run Agent ---
    final_answer = None
    reasoning_events = []
    error_message = None

    try:
        logger.info(f"Running agent for session {session_id}, task {task_id}...")
        async for event in runner.run_async(user_id=user_id, session_id=session_id, new_message=content):
            try:
                event_dict = event.model_dump(exclude_none=True, warnings=False)
            except Exception as dump_error:
                logger.warning(f"Could not serialize event {event.id}. Error: {dump_error}")
                event_dict = {"event_id": event.id, "author": event.author, "error": "Serialization failed"}
            reasoning_events.append(event_dict)

            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    full_response = event.content.parts[0].text.strip()
                    if "FINAL ANSWER:" in full_response:
                        answer_part = full_response.split("FINAL ANSWER:", 1)[-1].strip()
                        if answer_part.startswith("```") and answer_part.endswith("```"):
                           final_answer = answer_part[3:-3].strip()
                        elif answer_part.startswith("`") and answer_part.endswith("`"):
                           final_answer = answer_part[1:-1].strip()
                        else:
                           final_answer = answer_part
                    else:
                        logger.warning(f"Final response for task {task_id} did not contain 'FINAL ANSWER:'. Full response: {full_response}")
                        final_answer = full_response

        if final_answer is None and not error_message:
             logger.warning(f"Agent did not produce a discernible final answer for task {task_id}.")


    except Exception as e:
        logger.exception(f"Error running agent for task {task_id}: {e}") # 使用 logger.exception 包含 traceback
        error_message = f"An error occurred: {str(e)}"
        reasoning_events.append({"error": error_message, "traceback": traceback.format_exc()})


    # --- Return Response ---
    return ChatResponse(
        session_id=session_id,
        model_answer=final_answer,
        reasoning_trace=reasoning_events,
        error=error_message
    )

# --- Root Endpoint (Optional) ---
@app.get("/")
async def read_root():
    return {"message": "GAIA Solver Agent API is running."}

# --- Function for Uvicorn (if running directly) ---
# def run_api():
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     run_api()