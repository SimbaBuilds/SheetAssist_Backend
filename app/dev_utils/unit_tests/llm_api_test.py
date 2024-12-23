
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)


from app.utils.llm_service import LLMService

openai_client = LLMService().openai_client

system_prompt = "You are a helpful assistant."
user_content = "What is the capital of France?"

def openai_generate_text(system_prompt: str, user_content: str) -> str:
    """Generate text using OpenAI with system and user prompts"""
    response = openai_client.chat.completions.create(
        model="gpt4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    if not response.choices[0].message.content:
        raise ValueError("Empty response from API")
    return response.choices[0].message.content

print(openai_generate_text(system_prompt, user_content))