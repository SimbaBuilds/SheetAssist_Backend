import code
import threading
from io import StringIO
import sys
import contextlib
from typing import Optional
from app.schemas import SandboxResult
import ast
import matplotlib
import logging



# method to transform code to AST and capture last expression result 
def transform_ast(code: str) -> ast.AST:
    """Transform AST to capture last expression result"""
    logging.info("Starting AST transformation")
    try:
        tree = ast.parse(code)
        logging.info("Successfully parsed code into AST")
        
        if tree.body:
            last_node = tree.body[-1]
            logging.info(f"Last node type: {type(last_node).__name__}")
            
            if isinstance(last_node, ast.Expr):
                logging.info("Processing expression node")
                # Handle both single values and tuples/lists
                if isinstance(last_node.value, (ast.Tuple, ast.List)):
                    logging.info("Last node is a tuple/list")
                    value = last_node.value
                else:
                    logging.info("Creating tuple from single value")
                    value = ast.Tuple(elts=[last_node.value], ctx=ast.Load())
                
                assign = ast.Assign(
                    targets=[ast.Name(id="_result", ctx=ast.Store())],
                    value=value
                )
                tree.body[-1] = ast.fix_missing_locations(assign)
                logging.info("Successfully transformed expression node")
                
            elif isinstance(last_node, ast.Assign):
                logging.info("Processing assignment node")
                # Create a new assignment to _result that captures the target name
                result_assign = ast.Assign(
                    targets=[ast.Name(id="_result", ctx=ast.Store())],
                    value=ast.Name(id=last_node.targets[0].id, ctx=ast.Load()) if isinstance(last_node.targets[0], ast.Name) else last_node.targets[0]
                )
                tree.body.append(ast.fix_missing_locations(result_assign))
                logging.info("Successfully transformed assignment node")
            
            logging.info("AST transformation completed successfully")
            return tree
        else:
            logging.warning("Empty AST body")
            return tree
            
    except Exception as e:
        logging.error(f"Error during AST transformation: {str(e)}", exc_info=True)
        raise




class EnhancedPythonInterpreter:
    # Class constructor
    def __init__(self, timeout_seconds: Optional[int] = 60):
        # Add debug prints
        self.timeout_seconds = timeout_seconds
        self.setup_allowed_packages()
        self.setup_interpreter()
    
    def setup_allowed_packages(self):
        """Initialize allowed packages for the sandbox environment"""
        matplotlib.use('Agg')  # Set non-interactive backend
        import matplotlib.pyplot as plt
        plt.ioff()  # Disable interactive mode
        
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
            'matplotlib': matplotlib,
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
        
        logging.info("Starting code execution in sandbox")
        logging.debug(f"Code to execute:\n{code}")
        
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
            logging.info("Setting up restricted importer")
            sys.meta_path.insert(0, self.restricted_importer)
            try:
                logging.info("Starting output capture")
                with self.capture_output() as (stdout, stderr):
                    try:
                        logging.info("Starting AST transformation")
                        tree = transform_ast(code)
                        logging.info("AST transformation completed")
                        
                        ns = {
                            '__builtins__': self.safe_builtins,
                            **self.allowed_packages,
                            **(namespace or {})
                        }
                        logging.debug(f"Namespace keys: {list(ns.keys())}")
                        
                        logging.info("Compiling code")
                        compiled_code = compile(tree, '<ast>', 'exec')
                        logging.info("Executing compiled code")
                        
                        exec(compiled_code, ns)
                        logging.info("Code execution completed")
                        
                        if '_result' in ns:
                            logging.info(f"Result type: {type(ns.get('_result')).__name__}")
                            result.return_value = ns.get('_result')
                        else:
                            logging.warning("No '_result' found in namespace after execution")
                            
                        result.print_output = stdout.getvalue()
                        if stderr.getvalue():
                            stderr_output = stderr.getvalue()
                            logging.error(f"Stderr output: {stderr_output}")
                            result.error = stderr_output
                        logging.info("Code execution completed successfully")
                    except SyntaxError as e:
                        logging.error(f"Syntax error in code: {str(e)}", exc_info=True)
                        result.error = f"Syntax error: {str(e)}"
                    except Exception as e:
                        logging.error(f"Error during code execution: {str(e)}", exc_info=True)
                        result.error = str(e)
            finally:
                logging.info("Cleaning up restricted importer")
                if self.restricted_importer in sys.meta_path:
                    sys.meta_path.remove(self.restricted_importer)

        logging.info("Starting execution thread")
        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        logging.info("Waiting for thread completion")
        thread.join(timeout=self.timeout_seconds)

        if thread.is_alive():
            logging.error(f"Execution timed out after {self.timeout_seconds} seconds")
            result.timed_out = True
            result.error = f'Execution timed out after {self.timeout_seconds} seconds'
            thread = None
        else:
            logging.info("Thread completed execution")
        
        return result




