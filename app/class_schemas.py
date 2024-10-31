import pandas as pd

class TabularDataInfo:
    """Class for storing information about data being processed"""
    def __init__(self, df: pd.DataFrame = None, snapshot: str = None, 
                 data_type: str = None, file_name: str = None):
        self.df = df
        self.snapshot = snapshot
        self.data_type = data_type
        self.file_name = file_name


class SandboxResult:
    """Class for storing the result of a sandboxed code execution"""
    def __init__(self, original_query: str, print_output: str, code: str, error: str, return_value: str, timed_out: bool):
        self.original_query = original_query
        self.print_output = print_output
        self.code = code
        self.error = error
        self.return_value = return_value
        self.timed_out = timed_out  


class AnalysisResult:
    """Class for storing the result of an analysis of a sandboxed code execution"""
    def __init__(self, thinking: str, new_code: str):
        self.thinking = thinking
        self.new_code = new_code

