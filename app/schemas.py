from typing import Any, List, Optional
from pydantic import BaseModel


class OutputPreferences(BaseModel):
    type: str  # 'download' or 'online' - no longer Optional
    destination_url: Optional[str] = None
    format: Optional[str] = None  # One of: 'csv', 'xlsx', 'docx', 'txt', 'pdf'
    modify_existing: Optional[bool] = None

class FileMetadata(BaseModel):
    """Metadata about an uploaded file from frontend"""
    name: str
    type: str  # MIME type
    extension: str
    size: int
    index: int

class QueryRequest(BaseModel):
    web_urls: Optional[List[str]] = []
    files_metadata: Optional[List[FileMetadata]] = []
    query: str
    output_preferences: OutputPreferences  # no longer Optional



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

class TruncatedSandboxResult(BaseModel):
    original_query: str 
    print_output: Optional[str] = None
    error: Optional[str] = None
    timed_out: Optional[bool] = None
    return_value_snapshot: Optional[str] = None

class FileInfo(BaseModel):
    """Information about a downloadable file"""
    file_path: str
    media_type: str
    filename: str
    download_url: Optional[str] = None


class QueryResponse(BaseModel):
    """Unified response model for all query processing results"""
    result: TruncatedSandboxResult
    status: str  # "success" or "error"
    message: str  # Description of result or error message
    files: Optional[List[FileInfo]] = None  # For downloadable files
    num_images_processed: int = 0

    class Config:
        arbitrary_types_allowed = True


