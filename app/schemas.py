import pandas as pd
from typing import Union, Tuple, Any, List, Optional
from pydantic import BaseModel
from fastapi import UploadFile 





class QueryRequest(BaseModel):
    web_urls: Optional[List[str]] = []
    files: Optional[List[UploadFile]] = []
    query: str


class FileDataInfo:
    """Class for storing information about data being processed"""
    def __init__(self, content: Any = None, snapshot: str = None, 
                 data_type: str = None, original_file_name: str = None, new_file_path: str = None, url: str = None):
        self.content = content
        self.snapshot = snapshot
        self.data_type = data_type
        self.original_file_name = original_file_name
        self.new_file_path = new_file_path
        self.url = url

class SandboxResult:
    """Class for storing the result of a sandboxed code execution"""
    def __init__(self, original_query: str, print_output: str, code: str, 
                 error: str, return_value: Union[Tuple[Any, ...], Any], timed_out: bool,
                 return_value_snapshot: Optional[str] = None):
        self.original_query = original_query
        self.print_output = print_output
        self.code = code
        self.error = error
        self.return_value = return_value  # Can be a single value of any type or a tuple of any types
        self.timed_out = timed_out
        self.return_value_snapshot = return_value_snapshot



class ProcessedQueryResult(BaseModel):
    result: SandboxResult
    message: str





