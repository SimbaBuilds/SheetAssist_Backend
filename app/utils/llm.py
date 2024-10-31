from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from app.class_schemas import SandboxResult, AnalysisResult, TabularDataInfo
import pandas as pd
import json
from typing import Tuple

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# generate code from user query -- result goes to sandbox
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


# generate new code from error -- result goes to sandbox
def generate_new_code_from_error(result: SandboxResult) -> str:
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


# generate new code from analysis -- result goes to sandbox
def generate_new_code_from_analysis(result: SandboxResult, analysis_result: str) -> str:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of code that did not produce an error 
                but did not satisfy the user's original query and return a new script to try.
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in the 'df' variable as a pandas DataFrame.
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original user query, code, and LLM produced analysis:
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n
                Analysis:\n{analysis_result}"""}
        ]
    )
    return response.choices[0].message.content

# post-error result processing - processes random sample of result -- result goes to sentiment analysis function
def analyze_sandbox_result(result: SandboxResult, old_data: List[TabularDataInfo], new_data: TabularDataInfo) -> str:
    """Analyze the result of a sandboxed code execution and return an analysis"""
    
    old_data_snapshot = ""
    for data in old_data:
        old_data_snapshot += f"{data.file_name}:\n{data.snapshot}\n\n"
    
    new_data_snapshot = new_data.to_string()
        
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a successful sandboxed code execution and determine if the result would satisfy the user's original query.
                Do not include print statements -- ensure the last line returns the desired value.
                Respond in with either "yes, the result satisfies the user's original query" 
                or "no, the result does not satisfy the user's original query [one sentence explanation of how the result does not satisfy the user's original query]"
             """},
            {"role": "user", "content": f""" 
                Here is the original user query and snapshots of the new and old data:
                Original Query:\n{result.original_query}\n\n
                Old Data:\n{old_data_snapshot}\n\n
                New Data:\n{new_data_snapshot}\n\n
                """}
        ]
    )
    result = response.choices[0].message.content
    return result


# processes result of LLM analysis of post-error sandbox result -- result goes to generate new code or finish analysis
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
                "type": "object",
                "properties": {
                    "is_positive": {
                        "type": "boolean",
                        "description": "True if the sentiment is positive, False if negative"
                    }
                },
                "required": ["is_positive"],
                "additionalProperties": False
            },
            "strict": True
        },
        temperature=0
    )
    
    # Parse the JSON response and return the boolean
    result = json.loads(response.choices[0].message.content)
    return result["is_positive"], analysis_result

# Filter for error or result processing -- resutl goes to generate new code with error
def sandbox_result_filter_one(result: SandboxResult) -> AnalysisResult:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    if result.error:
        return generate_new_code(result)
    else:
        return analyze_sandbox_result(result)
