from app.schemas import SandboxResult, FileDataInfo
from app.utils.llm import gen_from_query, gen_from_error, gen_from_analysis, analyze_sandbox_result, sentiment_analysis
from app.utils.code_processing import extract_code
from typing import List
from app.utils.sandbox import EnhancedPythonInterpreter
import pandas as pd

def process_query(
    query: str, 
    sandbox: EnhancedPythonInterpreter,
    data: List[FileDataInfo] = None 
) -> SandboxResult:
    
    # Create execution namespace by extending base namespace
    namespace = dict(sandbox.base_namespace)  # Create a copy of base namespace
    
    # Handle different data types in namespace
    if data and len(data) > 0:
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            namespace[var_name] = file_data.content
            
            # Print information about the data
            if isinstance(file_data.content, pd.DataFrame):
                print(f"{var_name} shape:", file_data.content.shape)
            elif hasattr(file_data.content, '__len__'):
                print(f"{var_name} length:", len(file_data.content))
    
    try:
        # Initial code generation and execution
        suggested_code = gen_from_query(query, data)
        cleaned_code = extract_code(suggested_code)
        result = sandbox.execute_code(query, cleaned_code, namespace=namespace)  
        
        # Error handling for initial execution
        error_attempts = 1
        while result.error and error_attempts < 6:
            print("Error:", result.error)
            suggested_code = gen_from_error(result)
            unprocessed_llm_output = suggested_code 
            cleaned_code = extract_code(suggested_code)
            print("New code:", cleaned_code)
            result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
            error_attempts += 1
            print("Error attempt:", error_attempts)
            if error_attempts == 5:
                result.error = "Execution failed after 5 attempts"
                return result
            
        # Analysis and improvement loop
        analysis_attempts = 1
        while analysis_attempts < 6:    
            print("Analysis attempt:", analysis_attempts)
            old_data = data
            
            # Create new FileDataInfo based on return value type
            if isinstance(result.return_value, pd.DataFrame):
                new_data = FileDataInfo(
                    content=result.return_value,
                    snapshot=result.return_value.head(10).to_string(),
                    data_type="DataFrame",
                    original_file_name=data[0].original_file_name if data else None
                )
            else:
                new_data = FileDataInfo(
                    content=result.return_value,
                    snapshot=str(result.return_value)[:1000] if result.return_value is not None else None,
                    data_type=type(result.return_value).__name__,
                    original_file_name=data[0].original_file_name if data else None
                )

            #ADD MORE CASES^ ?    
            
            analysis_result = analyze_sandbox_result(result, old_data, new_data)
            success, analysis_result = sentiment_analysis(analysis_result)
            print("Analysis result:", analysis_result)
            if success:
                #SUCCESS
                print("\nSuccess!\n")
                print("Unprocessed LLM output:\n", unprocessed_llm_output) 
                result = SandboxResult(
                    original_query=query, 
                    print_output="", 
                    code=result.code, 
                    error=None, 
                    return_value=new_data.df, 
                    timed_out=False
                )
                return result 
            
            # Gen new code from analysis
            new_code = gen_from_analysis(result, analysis_result)
            unprocessed_llm_output = new_code 
            cleaned_code = extract_code(new_code)
            result = sandbox.execute_code(query, cleaned_code, namespace=namespace)

            # Restart error handling for new attempt 
            error_attempts = 1
            while result.error and error_attempts < 6:
                print("Error:", result.error)
                suggested_code = gen_from_error(result)
                cleaned_code = extract_code(suggested_code)
                print("New code:", cleaned_code)
                result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
                error_attempts += 1
                print("Error attempt:", error_attempts)
                if error_attempts == 5:
                    result.error = "Execution failed after 5 attempts"
                    return result      
                    
            analysis_attempts += 1
            print("Analysis attempt:", analysis_attempts)
            if analysis_attempts == 5:
                result.error = "Analysis failed after 5 attempts"
                return result

        return result
        
    except ConnectionError as e:
        result = SandboxResult(original_query=query, print_output="", code="", error=None, return_value=None, timed_out=False)
        error_details = f'Connection Error: {str(e)}'
        result.error = error_details
        return result
    except Exception as e:
        result = SandboxResult(original_query=query, print_output="", code="", error=None, return_value=None, timed_out=False)
        error_details = f'GPT interpretation failed: {str(e)}\nType: {type(e).__name__}'
        result.error = error_details
        return result
