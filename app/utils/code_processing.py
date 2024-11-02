import ast


# method to transform code to AST and capture last expression result 
def transform_ast(code: str) -> ast.AST:
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


# method to clean code -- removes language identifier and import statements
def extract_code(suggested_code: str) -> str:
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
