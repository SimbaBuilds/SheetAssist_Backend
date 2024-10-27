import code
import ast
import threading
from io import StringIO
import sys
import contextlib
from pydantic import create_model
from inspect import signature
from typing import Optional
from openai import OpenAI
import os

# Update client initialization
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class EnhancedPythonInterpreter:
    def __init__(self, openai_api_key: Optional[str] = None, timeout_seconds: int = 60):
        self.timeout_seconds = timeout_seconds
        # Update client initialization in constructor
        self.openai_client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.setup_interpreter()
        
    def setup_interpreter(self):
        self.safe_builtins = {
            'abs': abs, 'bool': bool, 'int': int, 'float': float,
            'str': str, 'list': list, 'dict': dict, 'set': set,
            'tuple': tuple, 'len': len, 'max': max, 'min': min,
            'print': print, 'range': range, 'round': round,
            'sum': sum, 'type': type
        }
        
        self.interpreter = code.InteractiveInterpreter(locals={
            '__builtins__': self.safe_builtins,
            '__name__': '__main__',
            '__doc__': None
        })

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

    def execute_code(self, code: str) -> dict:
        """Execute code with safety checks and timeout"""
        result = {
            'output': '',
            'error': None,
            'return_value': None,
            'timed_out': False
        }

        def execute():
            with self.capture_output() as (stdout, stderr):
                try:
                    # Transform AST to capture last expression
                    tree = self.transform_ast(code)
                    # Execute in restricted environment
                    ns = {'__builtins__': self.safe_builtins}
                    exec(compile(tree, '<ast>', 'exec'), ns)
                    result['return_value'] = ns.get('_result')
                    result['output'] = stdout.getvalue()
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

    async def interpret_query(self, query: str, use_gpt: bool = True) -> dict:
        """Interpret query using GPT (if available) or direct execution"""
        if not use_gpt or not self.openai_client:
            return self.execute_code(query)

        try:
            # Update API call to new syntax
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Generate Python code for the given query"},
                    {"role": "user", "content": query}
                ]
            )
            
            suggested_code = response.choices[0].message.content
            return self.execute_code(suggested_code)
            
        except Exception as e:
            return {
                'output': '',
                'error': f'GPT interpretation failed: {str(e)}',
                'return_value': None,
                'timed_out': False
            }

    def create_schema(self, func: callable) -> dict:
        """Create JSON schema for function parameters"""
        params = signature(func).parameters
        model = create_model(
            f"Input_{func.__name__}",
            **{name: (param.annotation, ... if param.default == param.empty else param.default)
               for name, param in params.items()}
        )
        return model.schema()
