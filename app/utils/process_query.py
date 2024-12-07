from app.schemas import SandboxResult, FileDataInfo
from app.utils.llm import gen_from_query, gen_from_error, gen_from_analysis, analyze_sandbox_result, sentiment_analysis
from app.utils.code_processing import extract_code
from app.utils.data_processing import get_data_snapshot, compute_dataset_diff, DatasetDiff, prepare_analyzer_context
from typing import List
from app.utils.sandbox import EnhancedPythonInterpreter
import pandas as pd
import logging

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
            # Preprocess DataFrame before adding to namespace
            namespace[var_name] = file_data.content
            
            # Print information about the data
            if isinstance(file_data.content, pd.DataFrame):
                print(f"{var_name} shape:", file_data.content.shape)
            elif hasattr(file_data.content, '__len__'):
                print(f"{var_name} length:", len(file_data.content))
        print(f"Namespace created for data {var_name}")
    try:
        # Initial code generation and execution
        suggested_code = gen_from_query(query, data)
        unprocessed_llm_output = suggested_code 
        cleaned_code = extract_code(suggested_code)
        result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
        print(f"\ncode executed with return value of type {type(result.return_value).__name__}\n")  
        
        # Error handling for initial execution
        error_attempts = 0
        past_errors = []
        while result.error and error_attempts < 6: #TODO: CHANGE TO 6 LATER
            print(f"\n\nError analysis {error_attempts}:")
            print(f"Error: {result.error}") #WHY ALWAYS EXECUTING THIS LOOP?
            past_errors.append(result.error)
            suggested_code = gen_from_error(result, error_attempts, data, past_errors)
            unprocessed_llm_output = suggested_code 
            cleaned_code = extract_code(suggested_code)
            print("New LLM output\n:", unprocessed_llm_output)
            result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
            print(f"\ncode executed with return value of type {type(result.return_value).__name__}\n")  
            print("Error:", result.error)
            error_attempts += 1
            if error_attempts == 6:  #TODO: CHANGE TO 6 LATER
                result.error = "Failed to interpret query. Please try rephrasing your request."
                return result
            
        # Analysis and improvement loop
        analysis_attempts = 1
        while analysis_attempts < 6:    
            logging.info(f"Starting post-error analysis attempt {analysis_attempts}")
            old_data = data
            
            # Create new FileDataInfo based on return value type (get_data_snapshot handles tuples)
            logging.info(f"Creating new FileDataInfo with return value of type {type(result.return_value).__name__}")
            new_data = FileDataInfo(
                content=result.return_value, #likely a tuple containing a dataframe or string
                snapshot=get_data_snapshot(result.return_value, type(result.return_value).__name__), 
                data_type=type(result.return_value).__name__,
                original_file_name=data[0].original_file_name if data else None
            )
            
            #dataset diff logic here
            analyzer_context = {}
            for i in range(len(old_data)):
                if isinstance(old_data[i].content, pd.DataFrame):
                    logging.info(f"Processing DataFrame from old_data[{i}]")
                    for j, item in enumerate(new_data.content):
                        if isinstance(item, pd.DataFrame):
                            diff_key = f"diff{i+1}_{j+1}"
                            logging.debug(f"Preparing analyzer context for {diff_key}")
                            analyzer_context[diff_key] = prepare_analyzer_context(old_data[i].content, item)

            logging.info("Analyzing sandbox result")
            analysis_result = analyze_sandbox_result(result, old_data, new_data, analyzer_context) 
            success, analysis_result = sentiment_analysis(analysis_result)
            logging.info(f"Analysis result: {analysis_result}")
            logging.info(f"Analysis success: {success}")
            if success:
                #SUCCESS
                print("\nSuccess!\n")
                print("Successful LLM output:\n", unprocessed_llm_output) 
                result = SandboxResult(
                    original_query=query, 
                    print_output="", 
                    code=result.code, 
                    error=None, 
                    return_value=new_data.content,
                    timed_out=False
                )
                return result 
            
            # Gen new code from analysis
            new_code = gen_from_analysis(result, analysis_result, data, past_errors)
            unprocessed_llm_output = new_code 
            print("New LLM output\n:", unprocessed_llm_output)
            cleaned_code = extract_code(new_code)
            result = sandbox.execute_code(query, cleaned_code, namespace=namespace)

            # Restart error handling for new attempt 
            error_attempts = 0
            while result.error and error_attempts < 6:
                print(f"\n\nError analysis {error_attempts}:")
                print("Error:", result.error)
                past_errors.append(result.error)
                suggested_code = gen_from_error(result, error_attempts, data, past_errors)
                unprocessed_llm_output = suggested_code 
                print("New LLM output\n:", unprocessed_llm_output)
                cleaned_code = extract_code(suggested_code)
                result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
                error_attempts += 1
                if error_attempts == 5:
                    result.error = "Failed to interpret query. Please try rephrasing your request."
                    return result      
                    
            analysis_attempts += 1
            print("Analysis attempt:", analysis_attempts)
            if analysis_attempts == 5:
                result.error = "Failed to interpret query. Please try rephrasing your request."
                return result

        return result
        
    except ConnectionError as e:
        result = SandboxResult(
            original_query=query,
            print_output="",
            code="",
            error=f'Connection Error: {str(e)}',
            return_value=None,
            timed_out=False
        )
        return result
    except Exception as e:
        result = SandboxResult(
            original_query=query,
            print_output="",
            code="",
            error=f'GPT interpretation failed: {str(e)}\nType: {type(e).__name__}',
            return_value=None,
            timed_out=False
        )
        return result
