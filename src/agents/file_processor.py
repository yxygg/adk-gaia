# src/agents/file_processor.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool # 确保导入 FunctionTool
from src.core.config import get_model
import logging

# 导入我们定义的工具函数
from src.tools.file_tools import (
    read_text_file,
    read_json_file,
    read_spreadsheet,
    read_docx_file,
    parse_pdb_file,
    extract_zip_content,
    process_pdf_with_gemini,
    process_audio_with_gemini,
)

logger = logging.getLogger(__name__)

FILE_PROCESSOR_MODEL = get_model("specialist_model_flash")

if not FILE_PROCESSOR_MODEL:
    raise ValueError("Model for File Processor Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool (移除 description 参数) ---
# ADK 会自动从 func 的 docstring 获取描述信息

read_text_tool = FunctionTool(
    func=read_text_file
    # description="Reads the entire content of a plain text file (.txt, .py, etc.)." #<-- REMOVE
)
read_json_tool = FunctionTool(
    func=read_json_file
    # description="Reads and parses a JSON (.json) or JSON Lines (.jsonl) file, returning its content as a string." #<-- REMOVE
)
read_spreadsheet_tool = FunctionTool(
    func=read_spreadsheet
    # description="Reads data from an Excel (.xlsx) or CSV (.csv) file. Optionally takes a 'query' string (pandas query syntax) to filter data. Returns data as markdown." #<-- REMOVE
)
read_docx_tool = FunctionTool(
    func=read_docx_file
    # description="Extracts text content from a Microsoft Word (.docx) document." #<-- REMOVE
)
parse_pdb_tool = FunctionTool(
    func=parse_pdb_file
    # description="Parses a Protein Data Bank (.pdb) file and returns a summary of its structure (chains, atoms)." #<-- REMOVE
)
extract_zip_tool = FunctionTool(
    func=extract_zip_content
    # description="Lists files within a ZIP archive (.zip) or extracts text content from a specified 'target_filename' within the archive." #<-- REMOVE
)
process_pdf_tool = FunctionTool(
    func=process_pdf_with_gemini
    # description="Processes a PDF document (.pdf) using a multimodal AI model. Takes a 'prompt' describing what to do (e.g., 'summarize', 'answer question based on this doc'). Returns the AI's text response." #<-- REMOVE
)
process_audio_tool = FunctionTool(
    func=process_audio_with_gemini
    # description="Processes an audio file (e.g., .mp3, .wav) using a multimodal AI model. Takes a 'prompt' describing what to do (e.g., 'transcribe', 'summarize the content'). Returns the AI's text response." #<-- REMOVE
)

# 定义 File Processor Agent
file_processor_agent = LlmAgent(
    name="FileProcessorAgent",
    model=FILE_PROCESSOR_MODEL,
    description=(
        "Specializes in reading, parsing, and extracting information from various file types "
        "including text, JSON, spreadsheets (Excel, CSV), documents (PDF, DOCX), PDB, and ZIP archives. "
        "Can also process PDF and audio using advanced AI."
    ),
    instruction=(
        "You are an expert file processor. Your goal is to process the file specified by the 'file_path' argument based on the user's 'prompt' or 'instruction'.\n"
        "1. **Identify File Type:** Determine the type of file from the 'file_path' extension.\n"
        "2. **Select Tool:** Choose the MOST appropriate tool from your available tools (check their descriptions from the function docstrings) to handle that file type and the requested action.\n" # 强调从 docstring 获取描述
        "   - Use `read_text_file` for plain text (.txt, .py).\n"
        "   - Use `read_json_file` for JSON/JSONL (.json, .jsonl).\n"
        "   - Use `read_spreadsheet` for Excel/CSV (.xlsx, .csv). If the user prompt implies filtering or specific data extraction, formulate a pandas `query` string for the tool.\n"
        "   - Use `read_docx_file` for Word documents (.docx).\n"
        "   - Use `parse_pdb_file` for Protein Data Bank files (.pdb).\n"
        "   - Use `extract_zip_content` for ZIP archives (.zip). If a specific file inside is mentioned, use the `target_filename` argument.\n"
        "   - Use `process_pdf_with_gemini` for PDF files (.pdf). Pass the user's core request as the 'prompt' argument to this tool.\n"
        "   - Use `process_audio_with_gemini` for audio files (.mp3, .wav). Pass the user's core request as the 'prompt' argument to this tool.\n"
        "3. **Execute Tool:** Call the selected tool with the 'file_path' and any other necessary arguments (like 'prompt', 'query', 'target_filename').\n"
        "4. **Return Result:** Return the 'content' or 'message' provided in the tool's output dictionary. If the tool reports an error, relay the error message."
    ),
    tools=[ # 列出所有可用的文件处理工具
        read_text_tool,
        read_json_tool,
        read_spreadsheet_tool,
        read_docx_tool,
        parse_pdb_tool,
        extract_zip_tool,
        process_pdf_tool,
        process_audio_tool,
    ],
)

logger.info(f"FileProcessorAgent initialized with model: {FILE_PROCESSOR_MODEL}")
logger.info(f"FileProcessorAgent Tools: {[tool.name for tool in file_processor_agent.tools]}")