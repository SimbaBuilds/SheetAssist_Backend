# Replace the current sys.path modifications with this more robust approach
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Now we can import from app
from app.utils.preprocessing import FilePreprocessor
from app.utils.llm_service import LLMService, AnthropicVisionProcessor
from app.schemas import FileUploadMetadata
from app.utils.s3_file_actions import S3PDFStreamer

import pytest
import os
import sys
from dotenv import load_dotenv
import logging
from supabase import create_client
from pathlib import Path
from pprint import pprint
import json
import boto3
from io import BytesIO
from PyPDF2 import PdfReader

pytestmark = pytest.mark.asyncio

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client directly
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Get test user ID from environment variable or set a test value
user_id = "695eadee-feda-492c-8f95-86f72fcc10c4"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Test file paths and S3 keys
LOCAL_TEXT_PDF = "/Users/cameronhightower/Programming Projects/SS_Assist_Backend/app/dev_utils/unit_tests/pdsat1_short.pdf"
S3_TEXT_PDF_KEY = "uploads/695eadee-feda-492c-8f95-86f72fcc10c4/1737851892495-k89cqhpv0ui-dsat1_reading.pdf"
S3_SCANNED_PDF_KEY = "uploads/695eadee-feda-492c-8f95-86f72fcc10c4/1737864226473-8qwyq3273d-Receipt_Batch_Short.pdf"

# Initialize AWS clients
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION', 'us-east-1')
)
S3_BUCKET = "sheet-assist-temp-bucket"

@pytest.fixture
def llm_service():
    return LLMService()

@pytest.fixture
def file_preprocessor():
    return FilePreprocessor(
        num_images_processed=0,
        supabase=supabase,
        user_id=user_id
    )

@pytest.fixture
def s3_pdf_metadata():
    """Create metadata for the S3 PDF file"""
    s3_response = s3_client.head_object(Bucket=S3_BUCKET, Key=S3_SCANNED_PDF_KEY)
    file_size = s3_response['ContentLength']
    
    return FileUploadMetadata(
        name="Receipt_Batch_Short.pdf",
        type="application/pdf",
        extension="pdf",
        size=file_size,
        s3_key=S3_SCANNED_PDF_KEY,
        index=0
    )

async def test_s3_file_access(s3_pdf_metadata):
    """Test if we can access the PDF file from S3"""
    print("\n=== Testing S3 File Access ===")
    
    try:
        # Initialize S3PDFStreamer
        streamer = S3PDFStreamer(s3_client, S3_BUCKET, s3_pdf_metadata.s3_key)
        
        # Try to get metadata about the PDF
        assert streamer.page_count > 0, "PDF should have at least one page"
        print(f"Successfully accessed PDF with {streamer.page_count} pages")
        
    except Exception as e:
        pytest.fail(f"Failed to access S3 file: {str(e)}")

async def test_pdf_page_streaming(s3_pdf_metadata):
    """Test if we can stream individual pages from the PDF"""
    print("\n=== Testing PDF Page Streaming ===")
    
    try:
        # Initialize S3PDFStreamer
        streamer = S3PDFStreamer(s3_client, S3_BUCKET, s3_pdf_metadata.s3_key)
        
        # Try to stream the first page
        page_data = streamer.stream_page(1)
        assert page_data is not None, "Page data should not be None"
        
        # Verify we can create a PDF reader from the page data
        reader = PdfReader(BytesIO(page_data))
        assert len(reader.pages) > 0, "Should be able to read the page"
        
        print("Successfully streamed and verified page content")
        
    except Exception as e:
        pytest.fail(f"Failed to stream PDF page: {str(e)}")

async def test_page_to_image_conversion(s3_pdf_metadata, llm_service):
    """Test if we can convert a PDF page to an image"""
    print("\n=== Testing PDF Page to Image Conversion ===")
    
    try:
        # Initialize S3PDFStreamer
        streamer = S3PDFStreamer(s3_client, S3_BUCKET, s3_pdf_metadata.s3_key)
        
        # Get the first page
        page_data = streamer.stream_page(1)
        reader = PdfReader(BytesIO(page_data))
        
        # Initialize vision processor
        vision_processor = AnthropicVisionProcessor(llm_service.anthropic_client)
        
        # Convert page to image
        img_bytes = await vision_processor._convert_pdf_page_to_image(reader.pages[0])
        
        assert img_bytes is not None, "Image conversion should produce bytes"
        assert len(img_bytes) > 0, "Image bytes should not be empty"
        print("Successfully converted PDF page to image")
        
    except Exception as e:
        pytest.fail(f"Failed to convert page to image: {str(e)}")

async def test_vision_processing(file_preprocessor, s3_pdf_metadata):
    """Test if we can process the image with vision API and get valid text content"""
    print("\n=== Testing Vision Processing ===")
    
    try:
        # Process the file with vision
        result = await file_preprocessor.preprocess_file(
            file=s3_pdf_metadata,
            query="Extract the text from this document",
            file_type='pdf',
            processed_data=[]  # Pass empty list instead of None
        )
        
        content, data_type, is_image_like = result
        
        print("\nVision Processing Results:")
        print(f"Data Type: {data_type}")
        print(f"Is Image-like: {is_image_like}")
        print(f"Content Length: {len(content)}")
        print("\nSample Content (first 500 chars):")
        print(f"{content[:500]}...")
        
        assert data_type == "vision_extracted"
        assert is_image_like == True
        assert isinstance(content, str)
        assert len(content) > 0
        
    except Exception as e:
        pytest.fail(f"Failed to process with vision: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])

# python -m pytest app/dev_utils/unit_tests/pdf_processing.py -v -s