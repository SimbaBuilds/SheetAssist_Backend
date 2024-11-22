import code
import threading
from io import StringIO
import sys
import contextlib
from typing import Optional
from app.schemas import SandboxResult
from app.utils.code_processing import transform_ast


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
            'pd': __import__('pandas'),
            'np': __import__('numpy'),
            'openpyxl': __import__('openpyxl'),
            'math': __import__('math'),
            'statistics': __import__('statistics'),
            'datetime': __import__('datetime').datetime,
            'json': __import__('json'),
            'csv': __import__('csv'),
            'PyPDF2': __import__('PyPDF2'),
            'fitz': __import__('fitz'),  # PyMuPDF for advanced PDF processing
            'io': __import__('io'),      # For StringIO/BytesIO operations
            're': __import__('re'),      # For regular expressions
            'chardet': __import__('chardet'),  # For character encoding detection
            'tabula': __import__('tabula'),  # For extracting tables from PDFs
            'zipfile': __import__('zipfile')  # For handling compressed files
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
            'system', 'subprocess',
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
                    'ssl', 'certifi', 'urllib3', 'http.client', 'socket',
                    'fitz', 'io', 're', 'chardet', 'tabula', 'zipfile'  # Added new packages
                }

            def find_spec(self, fullname, path, target=None):
                module_base = fullname.split('.')[0]
                if module_base in self.blacklist:
                    raise ImportError(f"Import of {fullname} is not allowed for security reasons")
                if module_base not in self.whitelist:
                    raise ImportError(f"Only whitelisted packages can be imported. {fullname} is not whitelisted.")
                return None

        sys.meta_path.insert(0, RestrictedImporter()) #updates import finder using sys.meta_path machinery


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

        result = SandboxResult(
            original_query=original_query,
            print_output="",
            code=code,
            error=None,
            return_value=None,
            timed_out=False,
            return_value_snapshot=None
        )

        def execute():
            with self.capture_output() as (stdout, stderr):
                try:
                    # Transform AST to capture last expression
                    tree = transform_ast(code)
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




