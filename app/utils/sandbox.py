import code
import ast
import threading
from io import StringIO
import sys
import contextlib
from typing import Optional
import os
from dotenv import load_dotenv
import pandas as pd
import fitz  # PyMuPDF
import numpy as np
import openpyxl  # for Excel support
from typing import Any, Optional, List
from app.utils.llm import gen_from_query, gen_from_error, gen_from_analysis, analyze_sandbox_result, sentiment_analysis
from app.class_schemas import TabularDataInfo, SandboxResult



class EnhancedPythonInterpreter:
    # Class constructor
    def __init__(self, timeout_seconds: Optional[int] = 60):
        # Add debug prints
        self.timeout_seconds = timeout_seconds
        self.setup_allowed_packages()
        self.setup_interpreter()
    
    def setup_allowed_packages(self):
        """Initialize allowed packages for the sandbox environment"""
        self.allowed_packages = {
            'pd': pd,
            'np': np,
            'openpyxl': openpyxl,
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            'datetime': __import__('datetime'),
            'json': __import__('json'),
            'csv': __import__('csv'),
            'PyPDF2': __import__('PyPDF2')
        }
    
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
        
        # Create base namespace with safe builtins and allowed packages
        self.base_namespace = {
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
            '__doc__': None,
            **self.allowed_packages  # Add allowed packages to namespace
        }
        
        # Initialize interpreter with base namespace
        self.interpreter = code.InteractiveInterpreter(locals=self.base_namespace)

        # Restricted import finder -- 
        class RestrictedImporter:
            def __init__(self):
                self.blacklist = {
                    'os', 'sys', 'subprocess', 'socket', 
                    'requests', 'urllib', 'http', 
                    'pathlib', 'shutil', 'tempfile'
                }
                self.whitelist = {
                    'pandas', 'numpy', 'PyMuPDF', 'openpyxl',
                    'datetime', 'json', 'csv', 'PyPDF2', 'pd', 'np', 'math', 'statistics',
                    'openai', 'anyio', 'anyio._backends', 'httpx', 'typing_extensions',
                    'ssl', 'certifi', 'urllib3', 'http.client', 'socket'
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
    def execute_code(self, original_query: str, code: str, namespace: dict = None) -> SandboxResult:
        """Execute (cleaned)code with safety checks and timeout"""

        result = SandboxResult(original_query=original_query, print_output="", code=code, error=None, return_value=None, timed_out=False)

        def execute():
            with self.capture_output() as (stdout, stderr):
                try:
                    # Transform AST to capture last expression
                    tree = self.transform_ast(code)
                    # Create namespace with safe builtins, allowed packages, and passed data
                    ns = {
                        '__builtins__': self.safe_builtins,
                        **self.allowed_packages,
                        **(namespace or {})  # Add the passed namespace if it exists
                    }
                    #run the code
                    exec(compile(tree, '<ast>', 'exec'), ns)
                    # Capture return value from the last expression in the AST
                    result.return_value = ns.get('_result')
                    # Captures print output
                    result.print_output = stdout.getvalue()
                    # Captures error output
                    if stderr.getvalue():
                        result.error = stderr.getvalue()
                except Exception as e:
                    result.error = str(e)

        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        print("thread started")
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            result.timed_out = True
            result.error = f'Execution timed out after {self.timeout_seconds} seconds'
            thread = None

        return result

    # method to clean code -- removes language identifier and import statements
    def extract_code(self, suggested_code: str) -> str:
        # Extract code enclosed in triple backticks
        code_start = suggested_code.find('```') + 3
        code_end = suggested_code.rfind('```')
        extracted_code = suggested_code[code_start:code_end].strip()
        # print("\nextracted_code:\n", extracted_code)
        # Remove language identifier if present
        if extracted_code.startswith('python'):
            extracted_code = extracted_code[6:].strip()

        # Remove import statements
        cleaned_code = '\n'.join(
            line for line in extracted_code.split('\n')
            if not line.strip().startswith('import') and not line.strip().startswith('from')
        )
        return cleaned_code

    # method to interpret query using GPT (if available) or direct execution -- wraps execute_code
    def process_query(self, query: str, use_gpt: bool = True, data: Optional[List[TabularDataInfo]] = None) -> SandboxResult:
        # Create execution namespace by extending base namespace
        namespace = dict(self.base_namespace)  # Create a copy of base namespace
        if data and len(data) > 0 and data[0].df is not None:
            namespace['df'] = data[0].df
            print("DataFrame shape:", data[0].df.shape)
        
        try:
            # Initial code generation and execution
            suggested_code = gen_from_query(query, data)
            cleaned_code = self.extract_code(suggested_code)
            result = self.execute_code(query, cleaned_code, namespace=namespace)  
            
            # Error handling for initial execution
            error_attempts = 1
            while result.error and error_attempts < 6:
                print("Error:", result.error)
                suggested_code = gen_from_error(result)
                cleaned_code = self.extract_code(suggested_code)
                print("New code:", cleaned_code)
                result = self.execute_code(query, cleaned_code, namespace=namespace)
                error_attempts += 1
                print("Error attempt:", error_attempts)
                if error_attempts == 5:
                    result.error = "Execution failed after 5 attempts"
                    return result
                
            # Analysis and improvement loop
            analysis_attempts = 1
            while analysis_attempts < 6:    
                print("Analysis attempt:", analysis_attempts)
                old_data = data
                new_data = TabularDataInfo(df=result.return_value, snapshot=result.return_value.head(10), file_name=data[0].file_name, data_type="DataFrame")
                analysis_result = analyze_sandbox_result(result, old_data, new_data)
                success, analysis_result = sentiment_analysis(analysis_result)
                print("Analysis result:", analysis_result)
                if success:
                    #SUCCESS
                    print("\nSuccess!\n")
                    result = SandboxResult(
                        original_query=query, 
                        print_output="", 
                        code=result.code, 
                        error=None, 
                        return_value=new_data.df, 
                        timed_out=False
                    )
                    return result 
                
                # Gen new code from analysis
                new_code = gen_from_analysis(result, analysis_result)
                cleaned_code = self.extract_code(new_code)
                result = self.execute_code(query, cleaned_code, namespace=namespace)

                # Restart error handling for new attempt 
                error_attempts = 1
                while result.error and error_attempts < 6:
                    print("Error:", result.error)
                    suggested_code = gen_from_error(result)
                    cleaned_code = self.extract_code(suggested_code)
                    print("New code:", cleaned_code)
                    result = self.execute_code(query, cleaned_code, namespace=namespace)
                    error_attempts += 1
                    print("Error attempt:", error_attempts)
                    if error_attempts == 5:
                        result.error = "Execution failed after 5 attempts"
                        return result      
                        
                analysis_attempts += 1
                print("Analysis attempt:", analysis_attempts)
                if analysis_attempts == 5:
                    result.error = "Analysis failed after 5 attempts"
                    return result

            return result
            
        except ConnectionError as e:
            result = SandboxResult(original_query=query, print_output="", code="", error=None, return_value=None, timed_out=False)
            error_details = f'Connection Error: {str(e)}'
            result.error = error_details
            return result
        except Exception as e:
            result = SandboxResult(original_query=query, print_output="", code="", error=None, return_value=None, timed_out=False)
            error_details = f'GPT interpretation failed: {str(e)}\nType: {type(e).__name__}'
            result.error = error_details
            return result



