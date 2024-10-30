from app.utils.sandbox import EnhancedPythonInterpreter
import asyncio


prompt = "Calculate the factorial of 5"

# Example usage
if __name__ == "__main__":
    interpreter = EnhancedPythonInterpreter()
    
    # # Direct code execution
    # result = interpreter.execute_code("print('Hello'); 2 + 2")
    # print("Direct execution:", result)
    
    # GPT-assisted interpretation
    result = asyncio.run(interpreter.interpret_query(
        prompt,
        use_gpt=True
    ))
    print("GPT-assisted query:", result)




#python -m app.main
