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
