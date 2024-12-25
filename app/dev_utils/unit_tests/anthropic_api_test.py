# Replace the current sys.path modifications with this more robust approach
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import pytest
import os
from app.utils.llm_service import LLMService
from pathlib import Path

@pytest.fixture
def vision_processor():
    """Create an instance of AnthropicVisionProcessor using environment API key"""
    return LLMService().anthropic_client

@pytest.fixture
def test_image(tmp_path):
    """Create a test image file"""
    image_path = tmp_path / "test_image.png"
    # Create a simple test image using PIL
    from PIL import Image, ImageDraw
    
    # Create a 200x200 white image
    img = Image.new('RGB', (200, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some text to the image
    draw.text((50, 100), "Hello, World!", fill='black')
    
    img.save(image_path)
    return str(image_path)

# @pytest.fixture
# def test_pdf(tmp_path):
#     """Create a test PDF file"""
#     from reportlab.pdfgen import canvas
    
#     pdf_path = tmp_path / "test.pdf"
#     c = canvas.Canvas(str(pdf_path))
#     c.drawString(100, 750, "Test PDF Document")
#     c.drawString(100, 700, "This is a test page.")
#     c.save()
    
#     return str(pdf_path)

def test_process_image_with_vision(vision_processor, test_image):
    """Test processing an image with Claude Vision"""
    query = "What text do you see in this image?"
    result = vision_processor.process_image_with_vision(test_image, query)
    
    assert result["status"] == "completed"
    assert isinstance(result["content"], str)
    assert len(result["content"]) > 0
    assert "Hello, World!" in result["content"].lower()

# def test_process_pdf_with_vision(vision_processor, test_pdf):
#     """Test processing a PDF with Claude Vision"""
#     query = "What text do you see in this PDF?"
#     result = vision_processor.process_pdf_with_vision(test_pdf, query)
    
#     assert result["status"] == "completed"
#     assert isinstance(result["content"], str)
#     assert len(result["content"]) > 0
#     assert "test pdf document" in result["content"].lower()

# def test_invalid_image_path(vision_processor):
#     """Test behavior with invalid image path"""
#     query = "What's in this image?"
#     result = vision_processor.process_image_with_vision("nonexistent.jpg", query)
    
#     assert result["status"] == "error"
#     assert "not found" in result["error"].lower()

# def test_invalid_pdf_path(vision_processor):
#     """Test behavior with invalid PDF path"""
#     query = "What's in this PDF?"
#     result = vision_processor.process_pdf_with_vision("nonexistent.pdf", query)
    
#     assert result["status"] == "error"
#     assert "not found" in result["error"].lower()

# def test_image_to_base64(vision_processor, test_image):
#     """Test converting image to base64"""
#     base64_string = vision_processor.image_to_base64(test_image)
#     assert isinstance(base64_string, str)
#     assert len(base64_string) > 0

# def test_pdf_page_to_base64(vision_processor, test_pdf):
#     """Test converting PDF page to base64"""
#     base64_string = vision_processor.pdf_page_to_base64(test_pdf, 0)
#     assert isinstance(base64_string, str)
#     assert len(base64_string) > 0
