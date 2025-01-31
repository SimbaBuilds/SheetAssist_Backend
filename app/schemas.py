from typing import Any, List, Optional
from pydantic import BaseModel
from datetime import datetime


class OutputPreferences(BaseModel):
    type: str  # 'download' or 'online' - no longer Optional
    destination_url: Optional[str] = None
    format: Optional[str] = None  # One of: 'csv', 'xlsx', 'docx', 'txt', 'pdf'
    modify_existing: Optional[bool] = None
    sheet_name: Optional[str] = None
    doc_name: Optional[str] = None




class FileUploadMetadata(BaseModel):
    """Metadata about an uploaded file from frontend"""
    name: str
    type: str  # MIME type
    extension: str
    size: int
    index: int
    file_id: Optional[str] = None  # Needed for batch processing file identification
    page_count: Optional[int] = None  # Number of pages in document
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None


class BatchProcessingFileInfo(BaseModel):
    """Information about a file during processing"""
    file_id: str
    page_range: tuple[int, int]
    metadata: FileUploadMetadata
    
    class Config:
        arbitrary_types_allowed = True  # For tuple type


class InputUrl(BaseModel):
    url: str
    sheet_name: Optional[str] = None
    doc_name: Optional[str] = None


class QueryRequest(BaseModel):
    input_urls: Optional[List[InputUrl]] = []
    files_metadata: Optional[List[FileUploadMetadata]] = []
    query: str
    output_preferences: OutputPreferences  # no longer Optional
    job_id: Optional[str] = None


class FileDataInfo(BaseModel):
    """Pydantic model for storing information about data being processed"""
    content: Optional[Any] = None #string or dataframe
    snapshot: Optional[str] = None
    data_type: Optional[str] = None
    original_file_name: Optional[str] = None
    new_file_path: Optional[str] = None
    url: Optional[str] = None
    metadata: Optional[dict] = None

    class Config:
        """docstring"""
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
    """Pydantic model for storing a truncated version of SandboxResult"""
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


# Handles batch and standard query processing results
class QueryResponse(BaseModel):
    """Unified response model for all query processing results"""
    original_query: Optional[str] = None    
    status: str  # "completed", "error", or "processing"
    message: str  # Description of result or error message
    files: Optional[List[FileInfo]] = None  # For downloadable files
    num_images_processed: int = 0
    job_id: Optional[str] = None  # Added for batch processing
    error: Optional[str] = None
    total_pages: Optional[int] = 0
    job_id: Optional[str] = None


class ChunkResponse(BaseModel):
    """Unified response model for all query processing results"""
    result: SandboxResult
    status: str  # "completed", "error", or "processing"
    message: str  # Description of result or error message
    files: Optional[List[FileInfo]] = None  # For downloadable files
    num_images_processed: int = 0
    job_id: Optional[str] = None  # Added for batch processing

    class Config:
        """docstring"""
        arbitrary_types_allowed = True

# jobs table
class Job(BaseModel):
    job_id: str
    user_id: str
    status: str  # "created", "processing", "completed", "error"
    total_pages: int
    processed_pages: int
    output_preferences: OutputPreferences
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_snapshot: Optional[dict] = None
    result_file_path: Optional[str] = None
    result_media_type: Optional[str] = None
    page_chunks: Optional[List[dict]] = None  
    current_chunk: Optional[int] = None       
    query: Optional[str] = None               
    message: Optional[str] = None           
    images_processed: Optional[List[dict]] = None  
    total_images_processed: Optional[int] = None  
    type: Optional[str] = None
    chunk_status: Optional[List[str]] = None
