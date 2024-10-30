from app.utils.sandbox import EnhancedPythonInterpreter
import asyncio
import pandas as pd
from app.utils.sandbox import TabularDataInfo


file_paths = ['course_data.csv', 'course_data_2.csv']

# Read the CSV file
df = pd.read_csv(file_paths[0])
# Store data info in a DataInfo object
data_info = TabularDataInfo(data=df, snapshot=str(df.head(10)), data_type="DataFrame", file_name=file_paths[0])


prompt = f"Remove courses with less than 20 active students from this list."



# Example usage
if __name__ == "__main__":
    interpreter = EnhancedPythonInterpreter()
    
    # # Direct code execution
    # result = interpreter.execute_code("print('Hello'); 2 + 2")
    # print("Direct execution:", result)
    
    # GPT-assisted interpretation
    result = asyncio.run(interpreter.interpret_query(
        prompt,
        use_gpt=True,
        data = [data_info]
    ))
    print("GPT-assisted query:", result)




#python -m app.main
