# src/agents/__init__.py
from .orchestrator import orchestrator_agent
from .file_processor import file_processor_agent # 导出新 Agent

# 可以选择性地导出其他 agent，如果需要直接访问的话
from .web_researcher import web_researcher_agent
from .code_executor import code_executor_agent

__all__ = [
    'orchestrator_agent',
    'file_processor_agent', # 添加到导出列表
    'web_researcher_agent',
    'code_executor_agent',
]