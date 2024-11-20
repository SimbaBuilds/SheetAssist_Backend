import pandas as pd
from typing import Union, Tuple, Any, List, Optional, Dict
from pydantic import BaseModel
from fastapi import UploadFile 
from datetime import datetime
from uuid import UUID



class OutputPreferences(BaseModel):
    type: Optional[str] = "online"  # 'download' or 'online'
    destination_url: Optional[str] = None
    format: Optional[str] = None  # One of: 'csv', 'xlsx', 'docx', 'txt', 'pdf'

class QueryRequest(BaseModel):
    web_urls: Optional[List[str]] = []
    files: Optional[List[UploadFile]] = []
    query: str
    output_preferences: Optional[OutputPreferences] = None


class FileInfo(BaseModel):
    """Information about a downloadable file"""
    file_path: str
    media_type: str
    filename: str

class QueryResponse(BaseModel):
    """Unified response model for all query processing results"""
    status: str  # "success" or "error"
    message: str  # Description of result or error message
    data: Optional[Any] = None  # For online viewing of results
    files: Optional[List[FileInfo]] = None  # For downloadable files

    class Config:
        arbitrary_types_allowed = True



class FileDataInfo(BaseModel):
    """Pydantic model for storing information about data being processed"""
    content: Optional[Any] = None
    snapshot: Optional[str] = None
    data_type: Optional[str] = None
    original_file_name: Optional[str] = None
    new_file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True  # Allow any Python type for content


class SandboxResult(BaseModel):
    """Pydantic model for storing the result of a sandboxed code execution"""
    original_query: str
    print_output: str
    code: str
    error: Optional[str] = None
    return_value: Any
    timed_out: bool
    return_value_snapshot: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allow any Python type for return_value




