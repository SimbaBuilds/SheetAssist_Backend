import pandas as pd
from typing import Union, Tuple, Any, List, Optional
from pydantic import BaseModel
from fastapi import UploadFile 





class QueryRequest(BaseModel):
    web_urls: Optional[List[str]] = []
    files: Optional[List[UploadFile]] = []
    query: str


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
    return_value: Any  # Can be a single value of any type or a tuple of any types
    timed_out: bool
    return_value_snapshot: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True  # Allow any Python type for return_value


class ProcessedQueryResult(BaseModel):
    result: SandboxResult
    message: str





