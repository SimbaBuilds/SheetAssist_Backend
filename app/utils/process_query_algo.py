from app.schemas import SandboxResult, FileDataInfo
from app.utils.llm_service import LLMService
from app.utils.data_processing import get_data_snapshot, compute_dataset_diff, DatasetDiff, prepare_analyzer_context
from typing import List, Tuple, Any, Optional
from app.utils.sandbox import EnhancedPythonInterpreter
import pandas as pd
import logging
import os
import json
from fastapi import Request
import numpy as np
from app.utils.connection_and_status import check_client_connection


def extract_code(suggested_code: str) -> str:
    # Extract code enclosed in triple backticks
    code_start = suggested_code.find('```') + 3
    code_end = suggested_code.rfind('```')
    extracted_code = suggested_code[code_start:code_end].strip()
    
    # Remove language identifier if present
    if extracted_code.startswith('python'):
        extracted_code = extracted_code[6:].strip()
    
    # Remove import statements and plt.show()
    cleaned_code = '\n'.join(
        line for line in extracted_code.split('\n')
        if not line.strip().startswith('import') and 
        not line.strip().startswith('from') and
        not line.strip() == 'plt.show()'
    )
    return cleaned_code

async def process_query_algo(
    request: Request,
    query: str, 
    sandbox: EnhancedPythonInterpreter,
    data: List[FileDataInfo] = None,
) -> SandboxResult:
    
    
    llm_service = LLMService()
    
    # Create execution namespace by extending base namespace
    namespace = dict(sandbox.base_namespace)
    

    # Handle different data types in namespace
    if data and len(data) > 0:
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            namespace[var_name] = file_data.content
            
            if isinstance(file_data.content, pd.DataFrame):
                print(f"{var_name} shape:", file_data.content.shape)
            elif hasattr(file_data.content, '__len__'):
                print(f"{var_name} length:", len(file_data.content))
        print(f"Namespace created for data {var_name}")

    try:
        # Initial code generation and execution
        provider, suggested_code = await llm_service.execute_with_fallback("gen_from_query", query, data)
        unprocessed_llm_output = suggested_code 
        cleaned_code = extract_code(suggested_code)
        result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
        print(f"\ncode executed with return value of type {type(result.return_value).__name__}\n")  
        
        # Error handling for initial execution
        error_attempts = 0
        past_errors = []
        while result.error and error_attempts < 6:
            print(f"\n\nError analysis {error_attempts}:")
            print(f"Error: {result.error}")
            past_errors.append(result.error)
            
            provider, suggested_code = await llm_service.execute_with_fallback(
                "gen_from_error", 
                result, 
                error_attempts, 
                data, 
                past_errors
            )
            
            unprocessed_llm_output = suggested_code 
            cleaned_code = extract_code(suggested_code)
            print("New LLM output\n:", unprocessed_llm_output)
            result = sandbox.execute_code(query, cleaned_code, namespace=namespace)
            error_attempts += 1
            if error_attempts == 6:

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
            
            # Prepare analyzer context
            full_diff_context = ""
            for i in range(len(old_data)):
                if isinstance(old_data[i].content, pd.DataFrame):
                    logging.info(f"Processing DataFrame from old_data[{i}]")
                    for j, item in enumerate(new_data.content):
                        if isinstance(item, pd.DataFrame):
                            this_context = {}
                            diff_key = f"diff{i+1}_{j+1}"
                            # Convert timestamps, NaT values, and other non-JSON serializable types
                            processed_item = item.copy()
                            # Handle datetime columns
                            for col in processed_item.select_dtypes(include=['datetime64[ns]']).columns:
                                processed_item[col] = processed_item[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                            # Handle other non-JSON serializable types
                            for col in processed_item.columns:
                                if processed_item[col].dtype == 'object':
                                    # Convert numpy types to native Python types
                                    processed_item[col] = processed_item[col].apply(lambda x: x.item() if hasattr(x, 'item') else x)
                            # Replace NaN, NaT, etc with None
                            processed_item = processed_item.replace({pd.NaT: None, pd.NA: None, np.nan: None})
                            
                            this_context[diff_key] = prepare_analyzer_context(old_data[i].content, processed_item)
                            logging.info(f"This context: {this_context}")
                            full_diff_context += json.dumps(this_context)
                            logging.info(f"Full diff context: {full_diff_context[:100]}...cont'd")
            # Analyze results
            provider, analysis_result = await llm_service.execute_with_fallback(
                "analyze_sandbox_result",
                result,
                old_data,
                new_data,
                full_diff_context
            )
            
            provider, (success, analysis_result) = await llm_service.execute_with_fallback(
                "sentiment_analysis",
                analysis_result
            )
            logging.info(f"Analysis result: {analysis_result}")
            print(f"\n\n----- Analysis result -----\n {analysis_result}\n")

            if success:
                #SUCCESS
                print("\nSuccess!\n")
                print("\n-------Successful LLM output:\n", unprocessed_llm_output, "\n-------") 
                result = SandboxResult(
                    original_query=query, 
                    print_output="", 
                    code=result.code, 
                    error=None, 
                    return_value=new_data.content,
                    timed_out=False
                )
                return result 
            
            # Generate new code from analysis
            provider, new_code = await llm_service.execute_with_fallback(
                "gen_from_analysis",
                result,
                analysis_result,
                data,
                past_errors
            )
            
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
                provider, suggested_code = await llm_service.execute_with_fallback(
                    "gen_from_error",
                    result,
                    error_attempts,
                    data,
                    past_errors
                )
                
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
        detailed_error = f'Connection Error: {str(e)}'
        result = SandboxResult(
            original_query=query,
            print_output="",
            code="",
            error=detailed_error,
            return_value=None,
            timed_out=False
        )
        return result
    except Exception as e:
        detailed_error = f'LLM interpretation failed: {str(e)}\nType: {type(e).__name__}'
        result = SandboxResult(
            original_query=query,
            print_output="",
            code="",
            error=detailed_error,
            return_value=None,
            timed_out=False
        )
        return result
