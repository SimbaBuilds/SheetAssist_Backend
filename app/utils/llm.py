from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from app.class_schemas import SandboxResult, FileDataInfo
import pandas as pd
import json
from typing import Tuple

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# generate code from user query -- result goes to sandbox
def gen_from_query(query: str, data: List[FileDataInfo]) -> str:
    
    user_message = query
    if data and len(data) > 0:
        # Build data description for multiple files
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\n{var_name} ({file_data.data_type}):\n{file_data.snapshot}\n"
        
        user_message = f"Available Data:{data_description}\n\nQuery: {query}"

    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Generate Python code for the given query. 
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in variables named 'data', 'data_1', 'data_2', etc.
                Each data variable may be of different types (DataFrame, string, list, etc.).
                The return value can be of any type (DataFrame, string, number, etc.).
                If you need to return multiple values, return them as a tuple: (value1, value2).
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": user_message}
        ]
    )
    
    if not response or not response.choices:
        raise ValueError("Empty response from OpenAI API")
            
    return response.choices[0].message.content

# generate new code from error -- result goes to sandbox
def gen_from_error(result: SandboxResult) -> str:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a failed sandboxed code execution and return a new script to try.
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in variables named 'data', 'data_1', 'data_2', etc.
                Each data variable may be of different types (DataFrame, string, list, etc.).
                The return value can be of any type (DataFrame, string, number, etc.).
                If you need to return multiple values, return them as a tuple: (value1, value2).
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original user query, code, and error:
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n\n
                Error:\n{result.error}"""}
        ]
    )
    return response.choices[0].message.content

# generate new code from analysis -- result goes to a_s_r
def gen_from_analysis(result: SandboxResult, analysis_result: str) -> str:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of code that did not produce an error 
                but did not satisfy the user's original query and return a new script to try.
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in variables named 'data', 'data_1', 'data_2', etc.
                Each data variable may be of different types (DataFrame, string, list, etc.).
                The return value can be of any type (DataFrame, string, number, etc.).
                If you need to return multiple values, return them as a tuple: (value1, value2).
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original user query, code, and LLM produced analysis:
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n
                Analysis:\n{analysis_result}"""}
        ]
    )
    return response.choices[0].message.content

# post-error result processing - processes random sample of result -- result goes to sentiment analysis function
def analyze_sandbox_result(result: SandboxResult, old_data: List[FileDataInfo], new_data: FileDataInfo) -> str:
    """Analyze the result of a sandboxed code execution and return an analysis"""
    
    # Build old data snapshot
    old_data_snapshot = ""
    for idx, data in enumerate(old_data):
        var_name = f'data_{idx}' if idx > 0 else 'data'
        old_data_snapshot += f"{var_name} ({data.data_type}):\n{data.snapshot}\n\n"
            
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a successful sandboxed code execution and determine if the result would satisfy the user's original query.
                The data can be of any type (DataFrame, string, list, etc.).
                Respond with either "yes, the result satisfies the user's original query" 
                or "no, the result does not satisfy the user's original query [one sentence explanation of how the result does not satisfy the user's original query]"
             """},
            {"role": "user", "content": f""" 
                Here is the original user query and snapshots of the new and old data:
                Original Query:\n{result.original_query}\n\n
                Old Data:\n{old_data_snapshot}\n\n
                New Data ({new_data.data_type}):\n{new_data.snapshot}\n\n
                """}
        ]
    )
    result = response.choices[0].message.content
    return result

# processes result of LLM analysis of post-error sandbox result -- result goes to generate new code or exit analysis
def sentiment_analysis(analysis_result: str) -> Tuple[bool, str]:
    """Analyze the sentiment of the result of an analysis and return a boolean"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
            You are a sentiment analyzer that evaluates text and determines if it has a positive sentiment.
            Return 'true' for positive sentiment (the result satisfies the user's original query) 
            and 'false' for negative sentiment (the result does not satisfy the user's original query).
            """},
            {"role": "user", "content": f"""Here is the result of an analysis:
                Analysis:\n{analysis_result}"""}
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "sentiment_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_positive": {
                            "type": "boolean",
                            "description": "True if the sentiment is positive, False if negative"
                        }
                    },
                    "required": ["is_positive"],
                    "additionalProperties": False,
                    "strict": True
                }
            }
        }
    )
    
    # Parse the JSON response and return the boolean
    result = json.loads(response.choices[0].message.content)
    return result["is_positive"], analysis_result


