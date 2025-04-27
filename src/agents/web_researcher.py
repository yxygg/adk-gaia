# src/agents/web_researcher.py
from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from src.core.config import get_model

WEB_RESEARCHER_MODEL = get_model("specialist_model_flash") # 使用 Flash 模型以提高效率

if not WEB_RESEARCHER_MODEL:
    raise ValueError("Model for Web Researcher Agent not found in configuration.")

web_researcher_agent = LlmAgent(
    name="WebResearcherAgent",
    model=WEB_RESEARCHER_MODEL,
    description="Specializes in searching the web using Google Search to find information based on queries.",
    instruction=(
        "You are an expert web researcher. Your primary tool is Google Search (`google_search`). "
        "Use this tool whenever a user query explicitly asks for web information, current events, "
        "or details that are likely found online and not part of general knowledge. "
        "Provide concise and relevant summaries based on the search results. "
        "If the search tool fails or returns no relevant information, state that clearly."
        "Focus ONLY on executing the search and returning the findings. Do not perform calculations or file operations."
    ),
    tools=[google_search] # 集成 ADK 内置的 Google Search 工具
)

print(f"WebResearcherAgent initialized with model: {WEB_RESEARCHER_MODEL}")