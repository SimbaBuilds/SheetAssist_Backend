import code
import ast
import threading
from io import StringIO
import sys
import contextlib
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
import pandas as pd
import fitz  # PyMuPDF
import numpy as np
import openpyxl  # for Excel support
from typing import Any, Optional, List


class TabularDataInfo:
    """Class for storing information about data being processed"""
    df: Optional[pd.DataFrame] = None  # The actual data object (DataFrame, array etc)
    snapshot: Optional[str] = None  # String representation of data preview/sample
    data_type: Optional[str] = None  # Type of data (e.g. "DataFrame", "ndarray", "list")
    file_name: Optional[str] = None  # Original file name if loaded from file



class EnhancedPythonInterpreter:
    # Class constructor
    def __init__(self, timeout_seconds: Optional[int] = 60):
        # Add debug prints
        load_dotenv(override=True)  # Add override=True to force reload
        api_key = os.getenv('OPENAI_API_KEY')        
        self.timeout_seconds = timeout_seconds
        self.openai_client = OpenAI(api_key=api_key)
        self.setup_allowed_packages()
        self.setup_interpreter()
    
    # method to define dangerous builtins
    def setup_interpreter(self):
        # Define dangerous builtin functions to remove
        dangerous_builtins = {
            # System operations
            'exec', 'eval', 'compile',
            # File operations
            'open', 
            # Process operations
            'system', 'subprocess', '__import__',
            # Object/memory operations
            'globals', 'locals', 'vars', 'memoryview',
            # Other dangerous operations
            'breakpoint', 'input'
        }
        
        # Start with a copy of regular builtins
        self.safe_builtins = dict(__builtins__) if isinstance(__builtins__, dict) else dict(vars(__builtins__))
        
        # Remove dangerous operations
        for func in dangerous_builtins:
            self.safe_builtins.pop(func, None)
        
        # Create interpreter with safe builtins and allowed packages
        namespace = {
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
            **self.allowed_packages  # Add allowed packages to namespace
        }
        self.interpreter = code.InteractiveInterpreter(locals=namespace)

        
        # Restricted import finder -- 
        class RestrictedImporter:
            def __init__(self):
                self.blacklist = {
                    'os', 'sys', 'subprocess', 'socket', 
                    'requests', 'urllib', 'http', 
                    'pathlib', 'shutil', 'tempfile'
                }
                # Whitelist for data manipulation packages
                self.whitelist = {
                    'pandas', 'numpy', 'PyMuPDF', 'openpyxl',
                    'datetime', 'json', 'csv', 'PyPDF2'
                }

            def find_spec(self, fullname, path, target=None):
                module_base = fullname.split('.')[0]
                if module_base in self.blacklist:
                    raise ImportError(f"Import of {fullname} is not allowed for security reasons")
                if module_base not in self.whitelist:
                    raise ImportError(f"Only whitelisted packages can be imported. {fullname} is not whitelisted.")
                return None

        sys.meta_path.insert(0, RestrictedImporter()) #updates import finder using sys.meta_path machinery

    # method to transform code to AST and capture last expression result 
    def transform_ast(self, code: str) -> ast.AST:
        """Transform AST to capture last expression result"""
        tree = ast.parse(code)
        if tree.body:
            last_node = tree.body[-1]
            if isinstance(last_node, ast.Expr):
                assign = ast.Assign(
                    targets=[ast.Name(id="_result", ctx=ast.Store())],
                    value=last_node.value
                )
                tree.body[-1] = ast.fix_missing_locations(assign)
        return tree

    # method to capture stdout and stderr (output and errors) -- decorator for use with Python "with" statement
    @contextlib.contextmanager
    def capture_output(self):
        """Capture stdout and stderr"""
        stdout, stderr = StringIO(), StringIO()
        old_stdout, old_stderr = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = stdout, stderr
            yield stdout, stderr
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr

    # method to execute raw code with safety checks and timeout
    def execute_code(self, code: str) -> dict:
        """Execute code with safety checks and timeout"""
        result = {
            'output': '',
            'code': code,
            'error': None,
            'return_value': None,
            'timed_out': False
        }

        def execute():
            with self.capture_output() as (stdout, stderr):
                try:
                    # Transform AST to capture last expression
                    tree = self.transform_ast(code)
                    # Update namespace to include allowed packages
                    ns = {
                        '__builtins__': self.safe_builtins,
                        **self.allowed_packages
                    }
                    #run the code
                    exec(compile(tree, '<ast>', 'exec'), ns)
                    # Capture return value from the last expression in the AST
                    result['return_value'] = ns.get('_result')
                    # Capture output
                    result['output'] = stdout.getvalue()
                    # Capture errors
                    if stderr.getvalue():
                        result['error'] = stderr.getvalue()
                except Exception as e:
                    result['error'] = str(e)

        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            result['timed_out'] = True
            result['error'] = f'Execution timed out after {self.timeout_seconds} seconds'
            thread = None

        return result

    
    # method to interpret query using GPT (if available) or direct execution -- wraps execute_code
    async def interpret_query(self, query: str, use_gpt: bool = True, data: Optional[List[DataInfo]] = None) -> dict:

        if not use_gpt or not self.openai_client:
            return self.execute_code(query)

        try:
            # Update API call to new syntax
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": """Generate Python code for the given query. 
                     The generated code should be enclosed in one set of triple backticks.  Don't forget your import statements.
                     Do not include print statements in the code -- instead, ensure the last line is an expression that returns the desired value.
                     For example, instead of `print(result)`, just use `result` as the last line of code.
                     """},
                    {"role": "user", "content": query + f"\n\n Here is a snapshot of the data: {data[0].snapshot}"}
                ]
            )
            
            suggested_code = response.choices[0].message.content
            # Extract code enclosed in triple backticks
            code_start = suggested_code.find('```') + 3
            code_end = suggested_code.rfind('```')
            extracted_code = suggested_code[code_start:code_end].strip()
            
            # Remove language identifier if present
            if extracted_code.startswith('python'):
                extracted_code = extracted_code[6:].strip()
            
            # Prepend common imports
            extracted_code = """import math\nimport statistics\nimport 
            datetime\nimport json\n import PyPDF2\nimport csv\nimport pandas as pd
            \nimport fitz\nimport numpy as np\nimport openpyxl\n\n""" + extracted_code

            return self.execute_code(extracted_code)  # returns dict
            
        except Exception as e:
            return {
                'output': '',
                'error': f'GPT interpretation failed: {str(e)}',
                'return_value': None,
                'timed_out': False
            }

