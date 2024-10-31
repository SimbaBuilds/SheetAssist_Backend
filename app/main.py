import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


from app.utils.sandbox import EnhancedPythonInterpreter
import pandas as pd
from app.utils.sandbox import TabularDataInfo


file_paths = ['course_data.csv', 'course_data_2.csv']
data_info_list = []

# Read the CSV file
df = pd.read_csv(file_paths[0])
# Store data info in a DataInfo object
data_info = TabularDataInfo(df=df, snapshot=str(df.head(10)), data_type="DataFrame", file_name=file_paths[0])
data_info_list.append(data_info)

query = "Remove courses with less than 20 active students from this list."



# Example usage
if __name__ == "__main__":
    interpreter = EnhancedPythonInterpreter()
    
    # # Direct code execution
    # result = interpreter.execute_code("print('Hello'); 2 + 2")
    # print("Direct execution:", result)
    
    # GPT-assisted interpretation
    result = interpreter.interpret_query(
        query = query,
        use_gpt=True,
        data = data_info_list
    )
    print("GPT-assisted query:", result)




#python -m app.main
