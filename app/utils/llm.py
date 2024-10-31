from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List
import time
import httpx

load_dotenv(override=True)


def generate_code(query: str, data: List) -> str:
    try:
        # Add debug print for API key (masked)
        api_key = os.getenv("OPENAI_API_KEY")
        print(f"API Key present: {'Yes' if api_key else 'No'}")
        
        client = OpenAI(
            api_key=api_key,
            timeout=60.0
        )

        # Verify API key is present
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        # Create user message with data context
        user_message = query
        if data and len(data) > 0:
            user_message = f"DataFrame Snapshot:\n{data[0].snapshot}\n\nQuery: {query}"

        print("Making API call...")
        response = client.chat.completions.create(
            model="gpt-4o",  
            messages=[
                {"role": "system", "content": """Generate Python code for the given query. 
                    The generated code should be enclosed in one set of triple backticks.
                    The data is available in the 'df' variable as a pandas DataFrame.
                    Do not include print statements -- ensure the last line returns the desired value."""},
                {"role": "user", "content": user_message}
            ]
        )
        
        if not response or not response.choices:
            raise ValueError("Empty response from OpenAI API")
                
        return response.choices[0].message.content

    except httpx.ConnectError as e:
        print(f"Detailed connection error: {str(e)}")  # Add detailed error logging
        raise ConnectionError(f"Connection error: {str(e)}")
    except httpx.TimeoutException as e:
        print(f"Timeout error: {str(e)}")  # Add timeout error logging
        raise ConnectionError(f"Timeout error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)} of type {type(e)}")  # Add unexpected error logging
        raise
        


