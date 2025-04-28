# src/agents/file_processor.py
import os
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from src.core.config import get_model
import logging

# 导入我们定义的工具函数
from src.tools.file_tools import (
    read_text_file,
    read_json_file,
    read_spreadsheet,
    read_docx_file,
    read_pptx_file,
    parse_pdb_file,
    extract_zip_content,
    process_pdf_with_gemini,
    process_audio_with_gemini,
    process_image_with_gemini
)

logger = logging.getLogger(__name__)

FILE_PROCESSOR_MODEL = get_model("specialist_model_flash")

if not FILE_PROCESSOR_MODEL:
    raise ValueError("Model for File Processor Agent not found in configuration.")

# --- 将函数包装成 ADK FunctionTool ---
read_text_tool = FunctionTool(func=read_text_file)
read_json_tool = FunctionTool(func=read_json_file)
read_spreadsheet_tool = FunctionTool(func=read_spreadsheet)
read_docx_tool = FunctionTool(func=read_docx_file)
read_pptx_tool = FunctionTool(func=read_pptx_file)
parse_pdb_tool = FunctionTool(func=parse_pdb_file)
extract_zip_tool = FunctionTool(func=extract_zip_content)
process_pdf_tool = FunctionTool(func=process_pdf_with_gemini)
process_audio_tool = FunctionTool(func=process_audio_with_gemini)
process_image_tool = FunctionTool(func=process_image_with_gemini)

# 定义 File Processor Agent
file_processor_agent = LlmAgent(
    name="FileProcessorAgent",
    model=FILE_PROCESSOR_MODEL,
    description=(
        "Specializes in reading, parsing, and extracting information from various file types "
        "including text, JSON, spreadsheets (Excel, CSV), documents (PDF, DOCX, PPTX), PDB, images (PNG, JPG), audio (MP3, WAV), and ZIP archives. " # 更新描述
        "Can process PDF, audio, and images using advanced AI."
    ),
    instruction=( # *** 修改指令以反映直接路径传递和新工具 ***
        "You are an expert file processor. You will receive a single string argument named `request` containing instructions and a file path.\n"
        "**IMPORTANT:** Your task is to parse the `request` string to extract the necessary information and then call the appropriate file processing tool.\n"
        "Follow these steps:\n"
        "1.  **Parse Request:** Carefully read the input `request` string.\n"
        "    - **Identify the File Path:** Extract the absolute file path mentioned in the request.\n"
        "    - **Identify the Action:** Determine the specific action requested (e.g., summarize, extract data, find specific info, list contents, transcribe).\n"
        "    - **Identify Optional Args:** Check if the request mentions a specific query for spreadsheets (look for 'using query ...') or a target filename for ZIP files (look for 'using target_filename ...').\n"
        "2.  **Determine File Type:** Based on the file path's extension, determine the file type.\n"
        "3.  **Select Tool:** Choose the MOST appropriate tool from your available tools based on the file type and the identified action.\n"
        "   - `read_text_file`: For .txt, .py, etc.\n"
        "   - `read_json_file`: For .json, .jsonl, .jsonld.\n"
        "   - `read_spreadsheet`: For .xlsx, .csv. Pass the extracted query string as the `query` argument if applicable.\n"
        "   - `read_docx_file`: For .docx.\n"
        "   - `read_pptx_file`: For .pptx.\n"
        "   - `parse_pdb_file`: For .pdb.\n"
        "   - `extract_zip_content`: For .zip. Pass the extracted target filename as `target_filename` if applicable.\n"
        "   - `process_pdf_with_gemini`: For .pdf. Pass the extracted action/prompt as the tool's `prompt` argument.\n"
        "   - `process_audio_with_gemini`: For audio files. Pass the extracted action/prompt as the tool's `prompt` argument.\n"
        "   - `process_image_with_gemini`: For image files. Pass the extracted action/prompt as the tool's `prompt` argument.\n"
        "4.  **Execute Tool:** Call the selected tool. Pass the extracted **file path** as the `file_path` argument. Pass any other extracted arguments (`prompt`, `query`, `target_filename`) to the corresponding tool parameters.\n"
        "5.  **Return Result:** Return the 'content' or 'message' from the tool's output dictionary. Relay any error messages accurately."
    ),
    tools=[ # 列出所有可用的文件处理工具
        read_text_tool,
        read_json_tool,
        read_spreadsheet_tool,
        read_docx_tool,
        read_pptx_tool,
        parse_pdb_tool,
        extract_zip_tool,
        process_pdf_tool,
        process_audio_tool,
        process_image_tool,
    ],
)

logger.info(f"FileProcessorAgent initialized with model: {FILE_PROCESSOR_MODEL}")
logger.info(f"FileProcessorAgent Tools: {[tool.name for tool in file_processor_agent.tools]}")