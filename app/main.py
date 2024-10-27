
from utils.agent import EnhancedPythonInterpreter



# Example usage
if __name__ == "__main__":
    interpreter = EnhancedPythonInterpreter(
        openai_api_key="your-api-key-here",
        timeout_seconds=5
    )
    
    # Direct code execution
    result = interpreter.execute_code("print('Hello'); 2 + 2")
    print("Direct execution:", result)
    
    # GPT-assisted interpretation
    import asyncio
    result = asyncio.run(interpreter.interpret_query(
        "Calculate the factorial of 5",
        use_gpt=True
    ))
    print("GPT-assisted:", result)