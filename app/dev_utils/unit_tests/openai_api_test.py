from openai import OpenAI
import os
from dotenv import load_dotenv
import sys

# Add the project root to Python path (going up two levels since we're in app/dev_utils)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from app.utils.llm import generate_code

print(generate_code("Hello", []))