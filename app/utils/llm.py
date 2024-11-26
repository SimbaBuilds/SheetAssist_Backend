from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
from app.schemas import SandboxResult, FileDataInfo
import json
from typing import Tuple

load_dotenv(override=True)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# generate code from user query -- result goes to sandbox
def gen_from_query(query: str, data: List[FileDataInfo]) -> str:
    
    if data and len(data) > 0:
        # Build data description for multiple files
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\nVariable Name: {var_name}\nData Type: {file_data.data_type}\nSnapshot:\n{file_data.snapshot}\n"
        

    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """ 
                You are a Python code generator that can read and process data from user provided data given a query.
                You are being given a preprocessed version of user provided files.
                The data will be of type DataFrame, string, list, etc. and is available in variables named 'data', 'data_1', 'data_2', etc...  
                Assume all data variables mentioned in the query already exist -- don't check for existence.
                The generated code should be enclosed in one set of triple backticks.
                Each data variable may be of different types (DataFrame, string, list, etc.).
                Don't try to concatenate to an empty or all-NA dataframe -- pandas no longer supports this.
                The return value can be of any type (DataFrame, string, number, etc.).
                If you need to return multiple values, return them as a tuple: (value1, value2).
                Do not forget your imports.
                Use the simplest method to return the desired value.
                Do not include print statements -- ensure the last line returns the desired value.
                If no further processing beyond preprocessing needs to be done, return the relevant data in the namespace variable(s). 
                Generate Python code for the given query and data.   
             """},
            {"role": "user", "content": f"Available Data:\n{data_description}\n\nQuery:\n{query}"}
        ]
    )
    print(f"LLM called with available data: {data_description} and query: {query} \nCode generated from query: \n {response.choices[0].message.content}")
    
    if not response or not response.choices:
        raise ValueError("Empty response from OpenAI API")
            
    return response.choices[0].message.content

# generate new code from error -- result goes to sandbox
def gen_from_error(result: SandboxResult, error_attempts: int, data: List[FileDataInfo]) -> str:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    
    if data and len(data) > 0:
    # Build data description for multiple files
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\nVariable Name: {var_name}\nData Type: {file_data.data_type}\nSnapshot:\n{file_data.snapshot}\n"

    
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a failed sandboxed code execution and return a new script to try.
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                The data is available in variables named 'data', 'data_1', 'data_2', etc.
                Each data variable may be of different types (DataFrame, string, list, etc.).
                Don't try to concatenate to an empty or all-NA dataframe -- pandas no longer supports this.
                The return value can be of any type (DataFrame, string, number, etc.).
                If you need to return multiple values, return them as a tuple: (value1, value2).
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original available data, user query, code, and error:
                Available Data:\n{data_description}\n\n
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n\n
                Error:\n{result.error}"""}
        ]
    )
    print(f"""Gen from error called -- attempt {error_attempts}, query: \n{result.original_query} 
          \ncode: \n{result.code} \nerror: \n{result.error}""")
    return response.choices[0].message.content

# generate new code from analysis -- result goes to a_s_r
def gen_from_analysis(result: SandboxResult, analysis_result: str, data: List[FileDataInfo]) -> str:
    """Analyze the result of a sandboxed code execution and return a new script to try"""
    
    if data and len(data) > 0:
    # Build data description for multiple files
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\nVariable Name: {var_name}\nData Type: {file_data.data_type}\nSnapshot:\n{file_data.snapshot}\n"    
    
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of the provided error free code that did not 
                satisfy the user's original query.  Then, return a new script to try.
                The data is available in variables named 'data', 'data_1', 'data_2', etc.
                Each data variable may be of different types (DataFrame, string, list, etc.).
                The return value can be of any type (DataFrame, string, number, etc.).
                If you need to return multiple values, return them as a tuple: (value1, value2).
                The generated code should be enclosed in one set of triple backticks.
                Do not forget your imports.
                Don't try to concatenate to an empty or all-NA dataframe -- pandas no longer supports this.
                Do not include print statements -- ensure the last line returns the desired value."""},
            {"role": "user", "content": f""" Here is the original user query, available data, code, and LLM produced analysis:
                Original Query:\n{result.original_query}\n\n
                Available Data:\n{data_description}\n\n
                Code:\n{result.code}\n\n
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
        old_data_snapshot += f"Variable name: {var_name}\nData type: {data.data_type}\nSnapshot:\n{data.snapshot}\n\n"
            
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=[
            {"role": "system", "content": """Analyze the result of a successful sandboxed code execution and determine if the result would satisfy the user's original query.
                File creation will be handled after this step: dataframes will later be converted to csv, xlsx etc... text will later be converted to txt, docx, etc... so do not judge based on return object type or whether a file was created.
                Respond with either "yes, the result seems to satisfy the user's query" 
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

def file_namer(query: str, data: List[FileDataInfo]) -> str:
    """Generate a suitable filename for the query result"""
    
    if data and len(data) > 0:
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\nVariable Name: {var_name}\nData Type: {file_data.data_type}\nSnapshot:\n{file_data.snapshot}\n"

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Using smaller model since this is a simple task
        messages=[
            {"role": "system", "content": """Generate a short, descriptive filename (without extension) for the data being processed.
                The filename should be:
                - Lowercase
                - Use underscores instead of spaces
                - Be descriptive but concise (max 3 underscore separated words)
                - Avoid special characters
                Return only the filename, nothing else."""},
            {"role": "user", "content": f"""Based on this query and data, suggest a filename:
                Query: {query}
                Available Data: {data_description}"""}
        ]
    )
    
    filename = response.choices[0].message.content.strip().lower()
    # Ensure filename is clean and safe
    filename = "".join(c for c in filename if c.isalnum() or c in ['_', '-'])
    return filename


