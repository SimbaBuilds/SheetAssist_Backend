import pandas as pd

class TabularDataInfo:
    """Class for storing information about data being processed"""
    def __init__(self, df: pd.DataFrame = None, snapshot: str = None, 
                 data_type: str = None, file_name: str = None):
        self.df = df
        self.snapshot = snapshot
        self.data_type = data_type
        self.file_name = file_name

