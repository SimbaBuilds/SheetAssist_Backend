from app.class_schemas import TabularDataInfo, SandboxResult
from app.utils.llm import gen_from_query, gen_from_error, gen_from_analysis, analyze_sandbox_result, sentiment_analysis
from app.utils.code_processing import extract_code
from typing import List
from app.utils.sandbox import EnhancedPythonInterpreter

def process_query(
    query: str, 
    sandbox: EnhancedPythonInterpreter,
    data: List[TabularDataInfo] = None #WILL NEED TO BE A LIST OF VARYING DATA TYPES
) -> SandboxResult:
    
    # Create execution namespace by extending base namespace
    namespace = dict(sandbox.base_namespace)  # Create a copy of base namespace
    if data and len(data) > 0 and data[0].df is not None:
        namespace['df'] = data[0].df
        print("DataFrame shape:", data[0].df.shape)
    
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
            new_data = TabularDataInfo(df=result.return_value, snapshot=result.return_value.head(10), file_name=data[0].file_name, data_type="DataFrame") 
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
