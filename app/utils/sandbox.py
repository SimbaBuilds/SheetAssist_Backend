import code
import threading
from io import StringIO
import sys
import contextlib
from typing import Optional
from app.schemas import SandboxResult
import ast



# method to transform code to AST and capture last expression result 
def transform_ast(code: str) -> ast.AST:
    """Transform AST to capture last expression result"""
    tree = ast.parse(code)
    if tree.body:
        last_node = tree.body[-1]
        if isinstance(last_node, ast.Expr):
            # Handle both single values and tuples/lists
            assign = ast.Assign(
                targets=[ast.Name(id="_result", ctx=ast.Store())],
                value=last_node.value if isinstance(last_node.value, (ast.Tuple, ast.List)) 
                      else ast.Tuple(elts=[last_node.value], ctx=ast.Load())
            )
            tree.body[-1] = ast.fix_missing_locations(assign)
        elif isinstance(last_node, ast.Assign):
            # Create a new assignment to _result that captures the target name
            result_assign = ast.Assign(
                targets=[ast.Name(id="_result", ctx=ast.Store())],
                value=ast.Tuple(elts=[last_node.targets[0]], ctx=ast.Load())
            )
            tree.body.append(ast.fix_missing_locations(result_assign))
    return tree




class EnhancedPythonInterpreter:
    # Class constructor
    def __init__(self, timeout_seconds: Optional[int] = 60):
        # Add debug prints
        self.timeout_seconds = timeout_seconds
        self.setup_allowed_packages()
        self.setup_interpreter()
    
    def setup_allowed_packages(self):
        """Initialize allowed packages for the sandbox environment"""
        import matplotlib.pyplot as plt  # Import pyplot properly
        
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
            'StringIO': __import__('io').StringIO,  # Add this line
            're': __import__('re'),      # For regular expressions
            'chardet': __import__('chardet'),  # For character encoding detection
            'tabula': __import__('tabula'),  # For extracting tables from PDFs
            
            # Core visualization packages
            'matplotlib': __import__('matplotlib'),
            'plt': plt,  # Use the properly imported pyplot
            'sns': __import__('seaborn'),  # Common alias for seaborn
            'Figure': __import__('matplotlib.figure'),
            'Axes': __import__('matplotlib.axes'),
            'seaborn': __import__('seaborn'),
            
            # Matplotlib components
            'colors': __import__('matplotlib.colors'),  # For color manipulation
            'cm': __import__('matplotlib.cm'),  # Color maps
            
            # Additional plotting utilities
            'mpl_toolkits': __import__('mpl_toolkits'),  # For 3D plotting
            'ticker': __import__('matplotlib.ticker'),  # For tick formatting
            'patches': __import__('matplotlib.patches'),  # For shapes and annotations
            'dates': __import__('matplotlib.dates'),  # For date formatting
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
        
        # Store the RestrictedImporter class as an instance variable
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
                    'fitz', 'io', 'StringIO', 're', 'chardet', 'tabula',
                    'matplotlib', 'matplotlib.pyplot', 'seaborn',
                    'matplotlib.figure', 'matplotlib.axes', 'matplotlib.colors',
                    'matplotlib.figure.Figure', 'matplotlib.axes.Axes',
                    'matplotlib.pyplot.subplots', 'matplotlib.gridspec',
                    'matplotlib.cm', 'mpl_toolkits', 'matplotlib.ticker',
                    'matplotlib.patches', 'matplotlib.dates'
                }

            def find_spec(self, fullname, path, target=None):
                module_base = fullname.split('.')[0]
                if module_base in self.blacklist:
                    raise ImportError(f"Import of {fullname} is not allowed for security reasons")
                if module_base not in self.whitelist:
                    raise ImportError(f"Only whitelisted packages can be imported. {fullname} is not whitelisted.")
                return None
        
        self.restricted_importer = RestrictedImporter()

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
            # Add and remove the restricted importer only during code execution
            sys.meta_path.insert(0, self.restricted_importer)
            try:
                with self.capture_output() as (stdout, stderr):
                    try:
                        tree = transform_ast(code)
                        ns = {
                            '__builtins__': self.safe_builtins,
                            **self.allowed_packages,
                            **(namespace or {})
                        }
                        exec(compile(tree, '<ast>', 'exec'), ns)
                        result.return_value = ns.get('_result')
                        result.print_output = stdout.getvalue()
                        if stderr.getvalue():
                            result.error = stderr.getvalue()
                    except Exception as e:
                        result.error = str(e)
            finally:
                # Always remove the restricted importer after execution
                if self.restricted_importer in sys.meta_path:
                    sys.meta_path.remove(self.restricted_importer)

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




