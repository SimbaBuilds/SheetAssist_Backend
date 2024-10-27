import ast
from pprint import pprint

def explain_ast(code: str):
    """Parse code into AST and explain its structure"""
    tree = ast.parse(code)
    print(f"\nCode:\n{code}")
    print("\nAST Structure:")
    print(ast.dump(tree, indent=2))
    return tree

# Example 1: Simple expression
print("=== Simple Expression ===")
explain_ast("x = 42")

# Example 2: Basic operation
print("\n=== Basic Operation ===")
explain_ast("a + b * c")

# Example 3: Function definition
print("\n=== Function Definition ===")
explain_ast("""
def greet(name):
    return "Hello, " + name
""")

# Example 4: AST Transformation Example
def transform_print_to_uppercase(node):
    """Transform all string literals in print statements to uppercase"""
    if isinstance(node, ast.Str):
        return ast.Str(s=node.s.upper())
    return node

class UppercasePrintTransformer(ast.NodeTransformer):
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            node.args = [transform_print_to_uppercase(arg) for arg in node.args]
        return node

# Example of AST transformation
print("\n=== AST Transformation Example ===")
original_code = """
print("hello, world")
print("this is a test")
"""

# Parse the code into an AST
tree = ast.parse(original_code)

# Transform the AST
transformer = UppercasePrintTransformer()
modified_tree = transformer.visit(tree)
ast.fix_missing_locations(modified_tree)

# Compile and execute the modified AST
print("Original code:")
print(original_code)
print("\nExecuting modified code:")
exec(compile(modified_tree, '<ast>', 'exec'))

# Example 5: AST Analysis
print("\n=== AST Analysis Example ===")

class CodeAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.function_count = 0
        self.variable_names = set()
        self.operation_count = 0
    
    def visit_FunctionDef(self, node):
        self.function_count += 1
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Store):
            self.variable_names.add(node.id)
        self.generic_visit(node)
    
    def visit_BinOp(self, node):
        self.operation_count += 1
        self.generic_visit(node)

# Analyze some code
code_to_analyze = """
def calculate_total(x, y):
    result = x + y
    return result

def multiply_numbers(a, b):
    product = a * b
    return product
"""

analyzer = CodeAnalyzer()
analyzer.visit(ast.parse(code_to_analyze))

print(f"Code to analyze:\n{code_to_analyze}")
print(f"\nAnalysis Results:")
print(f"Number of functions: {analyzer.function_count}")
print(f"Variable names: {analyzer.variable_names}")
print(f"Number of operations: {analyzer.operation_count}")