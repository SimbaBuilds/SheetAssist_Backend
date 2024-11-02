import pandas as pd
from typing import Union, BinaryIO, Tuple, Any


class SandboxResult:
    """Class for storing the result of a sandboxed code execution"""
    def __init__(self, original_query: str, print_output: str, code: str, 
                 error: str, return_value: Union[Tuple[Any, ...], Any], timed_out: bool):
        self.original_query = original_query
        self.print_output = print_output
        self.code = code
        self.error = error
        self.return_value = return_value  # Can be a single value of any type or a tuple of any types
        self.timed_out = timed_out


class AnalysisResult:
    """Class for storing the result of an analysis of a sandboxed code execution"""
    def __init__(self, thinking: str, new_code: str):
        self.thinking = thinking
        self.new_code = new_code


class TabularDataInfo:
    """Class for storing information about data being processed"""
    def __init__(self, df: pd.DataFrame = None, snapshot: str = None, 
                 data_type: str = None, file_name: str = None):
        self.df = df
        self.snapshot = snapshot
        self.data_type = data_type
        self.file_name = file_name


class TextDataInfo:
    """Class for storing information about text-based data (txt, docx)"""
    def __init__(self, content: str = None, file_name: str = None, 
                 data_type: str = None, original_format: str = None):
        self.content = content
        self.file_name = file_name
        self.data_type = data_type  # 'text'
        self.original_format = original_format  # 'txt' or 'docx'


class JsonDataInfo:
    """Class for storing information about JSON data"""
    def __init__(self, content: Union[dict, list, str] = None, 
                 string_representation: str = None, file_name: str = None):
        self.content = content  # Original JSON structure
        self.string_representation = string_representation  # Stringified version
        self.file_name = file_name
        self.data_type = 'json'


class ImageDataInfo:
    """Class for storing information about image data"""
    def __init__(self, image_data: BinaryIO = None, file_name: str = None,
                 original_format: str = None, converted_jpeg_path: str = None):
        self.image_data = image_data
        self.file_name = file_name
        self.original_format = original_format  # 'png' or 'jpeg'
        self.converted_jpeg_path = converted_jpeg_path  # Path to converted JPEG if original was PNG
        self.data_type = 'image'


class WebDataInfo:
    """Class for storing information about data from web URLs"""
    def __init__(self, url: str = None, df: pd.DataFrame = None,
                 source_type: str = None, converted_csv_path: str = None):
        self.url = url
        self.df = df  # Converted DataFrame
        self.source_type = source_type  # 'google_sheets' or 'excel_web'
        self.converted_csv_path = converted_csv_path
        self.data_type = 'web'




