from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from app.class_schemas import SandboxResult, AnalysisResult

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# generate code from user query
def generate_code(query: str, data: List) -> str:
    
    user_message = query
    if data and len(data) > 0:
        user_message = f"DataFrame Snapshot:\n{data[0].snapshot}\n\nQuery: {query}"

    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Generate Python code for the given query. 
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in the 'df' variable as a pandas DataFrame.
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": user_message}
        ]
    )
    
    if not response or not response.choices:
        raise ValueError("Empty response from OpenAI API")
            
    return response.choices[0].message.content



# sandbox result processing when error present
def generate_new_code(result: SandboxResult) -> str:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a failed sandboxed code execution and return a new script to try.
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in the 'df' variable as a pandas DataFrame.
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original user query, code, and error:
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n\n
                Error:\n{result.error}"""}
        ]
    )
    return response.choices[0].message.content



# result processing when no error - processes random sample of result
def analyze_sandbox_result(result: SandboxResult, data: List) -> AnalysisResult:
    """Analyze the result of a sandboxed code execution and return an analysis"""
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a successful sandboxed code execution and determine if the result would satisfy the user's original query.
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in the 'df' variable as a pandas DataFrame.
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original user query, code, and error:
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n\n
                Error:\n{result.error}"""}
        ]
    )
    result = AnalysisResult(thinking=response.choices[0].message.content, new_code=response.choices[1].message.content)
    return result


def analysis_triage(analysis_result: str) -> str:
    """Analyze the sentiment of the result of an analysis"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",  
        messages=[
            {"role": "system", "content": """Analyze the sentiment of the result of an analysis."""},
            {"role": "user", "content": f""" Here is the result of an analysis:
                Analysis:\n{analysis_result}"""}
        ]
    )
    return response.choices[0].message.content

# Filter for error or result processing
def sandbox_result_error_filter(result: SandboxResult) -> AnalysisResult:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    if result.error:
        return generate_new_code(result)
    else:
        return analyze_sandbox_result(result)
