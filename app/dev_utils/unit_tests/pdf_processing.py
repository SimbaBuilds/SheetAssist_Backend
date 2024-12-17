# Replace the current sys.path modifications with this more robust approach
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Now we can import from app
from app.utils.preprocessing import FilePreprocessor
from app.utils.llm_service import get_llm_service

import pytest
import os
import sys
from dotenv import load_dotenv
import logging
from supabase import create_client
from pathlib import Path
from pprint import pprint
import json

pytestmark = pytest.mark.asyncio

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client directly
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
llm_service = get_llm_service()

# Get test user ID from environment variable or set a test value
user_id = "695eadee-feda-492c-8f95-86f72fcc10c4"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test file paths
TEST_FILES_DIR = Path("app/dev_utils/unit_tests/test_pdfs")
READABLE_PDF = TEST_FILES_DIR / "long_readable.pdf"
UNREADABLE_PDF = TEST_FILES_DIR / "short_unreadable.pdf"
LONG_UNREADABLE_PDF = TEST_FILES_DIR / "long_unreadable.pdf"

@pytest.fixture
def llm_service():
    return get_llm_service()

@pytest.fixture
def file_preprocessor():
    return FilePreprocessor(
        num_images_processed=0,
        llm_service=llm_service,
        supabase=supabase,
        user_id=user_id
    )

async def test_machine_readable_pdf(file_preprocessor, llm_service):
    """Test processing of a machine-readable PDF"""
    
    print("\n=== Testing Machine Readable PDF ===")
    print(f"File: {READABLE_PDF}")
    
    with open(READABLE_PDF, 'rb') as file:
        result = await file_preprocessor.preprocess_file(
            file=file,
            file_type='pdf',
            llm_service=llm_service
        )
        
        content, data_type, is_readable = result
        
        print("\nMetadata:")
        print(f"Is Readable: {is_readable}")
        print(f"Data Type: {data_type}")
        print(f"Content Length: {len(content)}")
        print(f"Images Processed: {file_preprocessor.num_images_processed}")
        print("\nSample Content (first 500 chars):")
        print(f"{content[:500]}...")
        
        assert is_readable == True
        assert data_type == "text"
        assert isinstance(content, str)
        assert len(content) > 0
        assert file_preprocessor.num_images_processed == 0

# async def test_non_machine_readable_pdf(file_preprocessor, llm_service):
#     """Test processing of a non-machine-readable PDF (scanned document)"""
    
#     print("\n=== Testing Non-Machine Readable PDF ===")
#     print(f"File: {UNREADABLE_PDF}")
#     query = "Extract the text from this document"
#     print(f"Query: {query}")
    
#     with open(UNREADABLE_PDF, 'rb') as file:
#         result = await file_preprocessor.preprocess_file(
#             file=file,
#             file_type='pdf',
#             llm_service=llm_service
#         )
        
#         content, data_type, is_readable = result
        
#         print("\nMetadata:")
#         print(f"Is Readable: {is_readable}")
#         print(f"Data Type: {data_type}")
#         print(f"Content Length: {len(content)}")
#         print(f"Images Processed: {file_preprocessor.num_images_processed}")
#         print("\nSample Vision-Extracted Content (first 500 chars):")
#         print(f"{content[:500]}...")
        
#         assert is_readable == False
#         assert data_type == "vision_extracted"
#         assert isinstance(content, str)
#         assert len(content) > 0
#         assert file_preprocessor.num_images_processed > 0

# async def test_long_unreadable_pdf(file_preprocessor, llm_service):
#     """Test processing of a long unreadable PDF (should raise error)"""
    
#     print("\n=== Testing Long Unreadable PDF ===")
#     print(f"File: {LONG_UNREADABLE_PDF}")
    
#     with open(LONG_UNREADABLE_PDF, 'rb') as file:
#         try:
#             with pytest.raises(ValueError) as exc_info:
#                 await file_preprocessor.preprocess_file(
#                     file=file,
#                     file_type='pdf',
#                     llm_service=llm_service
#                 )
            
#             print("\nExpected Error Raised:")
#             print(str(exc_info.value))
            
#             assert "Unreadable PDF with more than 5 pages" in str(exc_info.value)
#         except Exception as e:
#             print(f"\nUnexpected Error: {str(e)}")
#             raise


# @pytest.mark.parametrize("query", [
#     "Summarize this document",
#     "What is the main topic?",
#     None
# ])
# async def test_pdf_with_different_queries(file_preprocessor, llm_service, query):
#     """Test PDF processing with different query types"""
    
#     print(f"\n=== Testing PDF with Query: {query} ===")
#     print(f"File: {UNREADABLE_PDF}")
    
#     with open(UNREADABLE_PDF, 'rb') as file:
#         result = await file_preprocessor.preprocess_file(
#             file=file,
#             file_type='pdf',
#             llm_service=llm_service
#         )
        
#         content, data_type, is_readable = result
        
#         print("\nMetadata:")
#         print(f"Query: {query}")
#         print(f"Is Readable: {is_readable}")
#         print(f"Data Type: {data_type}")
#         print(f"Content Length: {len(content)}")
#         print(f"Images Processed: {file_preprocessor.num_images_processed}")
#         print("\nSample Content (first 500 chars):")
#         print(f"{content[:500]}...")
        
#         assert isinstance(content, str)
#         assert len(content) > 0
#         assert not is_readable
#         assert data_type == "vision_extracted"

# if __name__ == "__main__":
#     pytest.main([__file__])



# python -m pytest app/dev_utils/unit_tests/pdf_processing.py -v -s