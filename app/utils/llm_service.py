from typing import Tuple, Any, Dict, List, Optional, Union, BinaryIO, AsyncGenerator
from openai import OpenAI
from anthropic import Anthropic
import logging
from app.schemas import SandboxResult, FileDataInfo
import json
from app.utils.system_prompts import gen_from_query_prompt, gen_from_query_prompt_image, gen_from_error_prompt, gen_from_analysis_prompt, analyze_sandbox_prompt, sentiment_analysis_prompt, file_namer_prompt, gen_visualization_prompt
import httpx
from PIL import Image
import io
import httpx
from openai import OpenAI
from app.schemas import FileDataInfo
import time
import base64
from pathlib import Path
import os
import fitz
import asyncio
from app.utils.s3_file_actions import s3_file_actions
from app.utils.s3_file_management import temp_file_manager
import gc
from PyPDF2 import PdfReader, PdfWriter
import tempfile
from pdf2image import convert_from_path
from io import BytesIO
from app.utils.s3_file_actions import S3PDFStreamer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Default values for environment variables
MAX_SNAPSHOT_LENGTH = int(os.getenv("MAX_SNAPSHOT_LENGTH", "2000"))
MAX_VISION_OUTPUT_TOKENS = int(os.getenv("MAX_VISION_OUTPUT_TOKENS", "3000"))
DEFAULT_LLM_PROVIDER = os.getenv("DEFAULT_LLM_PROVIDER", "anthropic")


def build_input_data_snapshot(input_data: List[FileDataInfo]) -> str:
    input_data_snapshot = ""
    for data in input_data:
        data_snapshot = data.snapshot
        input_data_snapshot += f"Original file name: {data.original_file_name}\nData type: {data.data_type}\nData Snapshot:\n{data_snapshot}\n\n"
    return input_data_snapshot


class BaseVisionProcessor:
    """Base class for vision processing with common PDF handling logic"""
    
    async def process_pdf_with_vision(
        self, 
        pdf_path: str = None,
        s3_key: str = None,
        query: str = None, 
        input_data: List[FileDataInfo] = None,
        page_range: Optional[tuple[int, int]] = None,
        use_s3: bool = False,
        stream: BinaryIO = None
    ) -> Dict[str, str]:
        """Process PDF with Vision API - common logic for both providers"""
        try:
            if use_s3 and s3_key:
                result = await self._process_s3_pdf(s3_key, query, input_data, page_range)
            elif stream:
                result = await self._process_stream_pdf(stream, query, input_data, page_range)
            elif pdf_path:
                result = await self._process_local_pdf(pdf_path, query, input_data, page_range)
            else:
                raise ValueError("Either pdf_path, s3_key, or stream must be provided")

            # If no content was extracted from any page
            if not result or (isinstance(result, dict) and not result.get("content")):
                return {
                    "status": "error",
                    "error": "No content could be extracted from the PDF"
                }

            return result

        except Exception as e:
            logger.error(f"Error in process_pdf_with_vision: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def _process_s3_pdf(self, s3_key: str, query: str, input_data: List[FileDataInfo], page_range: Optional[tuple[int, int]]) -> Dict[str, str]:
        """Process PDF stored in S3 one page at a time"""
        all_page_content = []
        input_data_snapshot = build_input_data_snapshot(input_data)
        
        try:
            # Initialize S3PDFStreamer
            streamer = S3PDFStreamer(s3_file_actions.s3_client, s3_file_actions.bucket, s3_key)
            
            # Determine page range
            start_page = int(page_range[0]) if page_range else 1
            end_page = min(int(page_range[1]), streamer.page_count) if page_range else streamer.page_count
            
            for page_num in range(start_page, end_page + 1):
                try:
                    # Get page and convert to image
                    page_data = streamer.stream_page(page_num)
                    reader = PdfReader(BytesIO(page_data))
                    
                    # Create a temporary file for the page image
                    with io.BytesIO() as img_buffer:
                        # Convert page to image
                        img_bytes = await self._convert_pdf_page_to_image(reader.pages[0])
                        if not img_bytes:
                            logger.error(f"Failed to convert page {page_num} to image")
                            continue
                            
                        img_buffer.write(img_bytes)
                        img_buffer.seek(0)
                        b64_page = base64.b64encode(img_buffer.read()).decode()
                    
                    # Process with provider-specific vision API
                    page_content = await self._process_single_page(
                        b64_page,
                        query,
                        input_data_snapshot,
                        page_num,
                        streamer.page_count
                    )
                    
                    if page_content:
                        all_page_content.append(f"[Page {page_num} of {streamer.page_count}]\n{page_content}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num}: {str(e)}")
                    continue
                    
                finally:
                    # Force garbage collection after each page
                    gc.collect()
                    await asyncio.sleep(float(os.getenv("SLEEP_TIME")))
            
            if not all_page_content:
                return {
                    "status": "error",
                    "error": "Failed to extract content from any pages"
                }
            
            return {
                "status": "completed",
                "content": "\n\n".join(all_page_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF from S3: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def _process_local_pdf(self, pdf_path: str, query: str, input_data: List[FileDataInfo], page_range: Optional[tuple[int, int]]) -> Dict[str, str]:
        """Process local PDF file one page at a time"""
        if not os.path.exists(pdf_path):
            raise ValueError(f"PDF file not found at path: {pdf_path}")
            
        stream = None
        doc = None
        all_page_content = []
        
        input_data_snapshot = build_input_data_snapshot(input_data)
        
        try:
            # Open file as stream
            stream = open(pdf_path, 'rb')
            doc = PdfReader(stream)
            
            # Determine page range
            start_page = int(page_range[0]) if page_range else 0
            end_page = min(int(page_range[1]), len(doc.pages)) if page_range else len(doc.pages)
            
            for page_num in range(start_page, end_page):
                try:
                    # Get page and convert to image
                    page = doc.pages[page_num]
                    
                    # Create a temporary file for the page image
                    with io.BytesIO() as img_buffer:
                        # Convert page to image
                        img_bytes = await self._convert_pdf_page_to_image(page)
                        img_buffer.write(img_bytes)
                        img_buffer.seek(0)
                        b64_page = base64.b64encode(img_buffer.read()).decode()
                    
                    # Process with provider-specific vision API
                    page_content = await self._process_single_page(
                        b64_page,
                        query,
                        input_data_snapshot,
                        page_num + 1,
                        len(doc.pages)
                    )
                    
                    all_page_content.append(f"[Page {page_num + 1} of {len(doc.pages)}]\n{page_content}")
                    
                finally:
                    # Clean up page resources immediately
                    if page:
                        page = None
                    # Force garbage collection after each page
                    gc.collect()
                    await asyncio.sleep(float(os.getenv("SLEEP_TIME")))
        
        finally:
            # Clean up resources
            if stream:
                stream.close()
        
        return {
            "status": "completed",
            "content": "\n\n".join(all_page_content)
        }

    async def _process_stream_pdf(self, stream: BinaryIO, query: str, input_data: List[FileDataInfo], page_range: Optional[tuple[int, int]]) -> Dict[str, str]:
        """Process PDF from a stream one page at a time"""
        all_page_content = []
        input_data_snapshot = build_input_data_snapshot(input_data)
        doc = None
        
        try:
            # Open PDF stream with PyPDF2
            doc = PdfReader(stream)
            total_pages = len(doc.pages)
            
            # Determine page range
            start_page = int(page_range[0]) if page_range else 0
            end_page = min(int(page_range[1]), total_pages) if page_range else total_pages
            
            for page_num in range(start_page, end_page):
                try:
                    # Get page and convert to image
                    page = doc.pages[page_num]
                    
                    # Create a temporary file for the page image
                    with io.BytesIO() as img_buffer:
                        # Convert page to image
                        img_bytes = await self._convert_pdf_page_to_image(page)
                        if not img_bytes:
                            logger.error(f"Failed to convert page {page_num + 1} to image")
                            continue
                            
                        img_buffer.write(img_bytes)
                        img_buffer.seek(0)
                        b64_page = base64.b64encode(img_buffer.read()).decode()
                    
                    # Process with provider-specific vision API
                    page_content = await self._process_single_page(
                        b64_page,
                        query,
                        input_data_snapshot,
                        page_num + 1,
                        total_pages
                    )
                    
                    if page_content:
                        all_page_content.append(f"[Page {page_num + 1} of {total_pages}]\n{page_content}")
                    
                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue
                    
                finally:
                    # Clean up page resources immediately
                    if page:
                        page = None
                    # Force garbage collection after each page
                    gc.collect()
                    await asyncio.sleep(float(os.getenv("SLEEP_TIME")))
            
            if not all_page_content:
                return {
                    "status": "error",
                    "error": "Failed to extract content from any pages"
                }
            
            return {
                "status": "completed",
                "content": "\n\n".join(all_page_content)
            }
            
        except Exception as e:
            logger.error(f"Failed to process PDF from stream: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    async def _convert_pdf_page_to_image(self, page) -> bytes:
        """Convert a PDF page to an image. To be implemented by provider-specific classes."""
        raise NotImplementedError

    async def _process_single_page(self, b64_page: str, query: str, input_data_snapshot: str, page_number: int, total_pages: int) -> str:
        """Process a single page - to be implemented by provider-specific classes"""
        raise NotImplementedError


class OpenaiVisionProcessor(BaseVisionProcessor):
    def __init__(self, openai_client: OpenAI = None):
        self.client = openai_client

    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def process_image_with_vision(
        self, 
        image_path: str = None,
        s3_key: str = None,
        query: str = None, 
        input_data: List[FileDataInfo] = None,
        use_s3: bool = False
    ) -> dict:
        """Process image with GPT-4 Vision API"""
        try:
            if use_s3 and s3_key:
                # Get metadata to get content length
                metadata = await temp_file_manager.get_file_metadata(s3_key)
                if not metadata:
                    raise ValueError(f"Could not get metadata for S3 key: {s3_key}")
                content_length = int(metadata.get('ContentLength', 0))
                # Get entire file content
                image_data = await s3_file_actions.get_file_range(s3_key, 0, content_length - 1)
                b64_image = base64.b64encode(image_data).decode()
            else:
                # Check if image_path exists
                if not os.path.exists(image_path):
                    raise ValueError(f"Image file not found at path: {image_path}")
                    
                b64_image = self.image_to_base64(image_path)
            
            input_data_snapshot = build_input_data_snapshot(input_data)
            
            completion = self.client.chat.completions.create(
                model=os.getenv("OPENAI_MAIN_MODEL"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                Your job is to extract relevant information from an image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).  Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_VISION_OUTPUT_TOKENS
            )
            print(f"\n ------- LLM called with query: {query} and input data snapshot: {input_data_snapshot} ------- \n")
            return {
                "status": "completed",
                "content": completion.choices[0].message.content
            }

        except httpx.ConnectError as e:
            return {
                "status": "error",
                "error": "Connection error to OpenAI service",
                "detail": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _convert_pdf_page_to_image(self, page) -> bytes:
        """Convert a PDF page to an image using pdf2image"""
        try:
            # Create a temporary file for the page
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                # Write the page to a temporary PDF
                writer = PdfWriter()
                writer.add_page(page)
                writer.write(temp_pdf)
                temp_pdf.flush()
                
                # Convert the page to image
                images = convert_from_path(temp_pdf.name, dpi=200, fmt='jpeg')
                
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='JPEG', quality=95)
                img_byte_arr.seek(0)
                
                return img_byte_arr.getvalue()
        finally:
            # Clean up temporary file
            if 'temp_pdf' in locals():
                os.unlink(temp_pdf.name)

    async def _process_single_page(self, b64_page: str, query: str, input_data_snapshot: str, page_number: int, total_pages: int) -> str:
        try:
            # Convert base64 to image bytes
            image_bytes = base64.b64decode(b64_page)
            
            # Create PIL Image from bytes
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG to a new bytes buffer
                jpeg_buffer = io.BytesIO()
                img.save(jpeg_buffer, format='JPEG', quality=95)
                jpeg_buffer.seek(0)
                
                # Convert JPEG bytes to base64
                jpeg_b64 = base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')

            completion = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=os.getenv("OPENAI_MAIN_MODEL"),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""Your job is to extract relevant information from this pdf image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).
                                Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
                                """
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{jpeg_b64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=MAX_VISION_OUTPUT_TOKENS
            )
            
            if not completion or not completion.choices:
                logger.error("No completion returned from OpenAI")
                return None
                
            page_content = completion.choices[0].message.content
            print(f"""\n -------LLM called with query: {query} on page: {page_number} of {total_pages} ------- \n\n
            Input data snapshot:\n {input_data_snapshot}
            Page Content:\n {page_content}\n
            """)
            
            return page_content
            
        except Exception as e:
            logging.error(f"Error processing PDF page with vision: {str(e)}")
            return None


class AnthropicVisionProcessor(BaseVisionProcessor):
    def __init__(self, anthropic_client: Anthropic = None):
        self.client = anthropic_client

    def image_to_base64(self, image_path: str) -> str:
        """Convert image file to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    async def process_image_with_vision(
        self, 
        image_path: str = None,
        s3_key: str = None,
        query: str = None, 
        input_data: List[FileDataInfo] = None,
        use_s3: bool = False
    ) -> dict:
        """Process image with Claude 3 Vision API"""
        try:
            if use_s3 and s3_key:
                # Get metadata to get content length
                metadata = await temp_file_manager.get_file_metadata(s3_key)
                logger.info(f"metadata: {metadata}")
                if not metadata:
                    raise ValueError(f"Could not get metadata for S3 key: {s3_key}")
                content_length = int(metadata.get('ContentLength', 0))
                # Get entire file content
                image_data = await s3_file_actions.get_file_range(s3_key, 0, content_length - 1)
                image_data_b64 = base64.b64encode(image_data).decode("utf-8")
                # Determine media type from S3 key
                media_type = f"image/{Path(s3_key).suffix[1:].lower()}"
                if media_type == "image/jpg":
                    media_type = "image/jpeg"
            else:
                # Check if image_path exists
                if not os.path.exists(image_path):
                    raise ValueError(f"Image file not found at path: {image_path}")
                    
                with open(image_path, "rb") as image_file:
                    image_data_b64 = base64.b64encode(image_file.read()).decode("utf-8")
                
                # Determine media type based on file extension
                media_type = f"image/{Path(image_path).suffix[1:].lower()}"
                if media_type == "image/jpg":
                    media_type = "image/jpeg"

            input_data_snapshot = build_input_data_snapshot(input_data)
            
            message = self.client.messages.create(
                model=os.getenv("ANTHROPIC_MAIN_MODEL"),
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": image_data_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""Your job is to extract relevant information from an image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).
                                Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
                                """
                            }
                        ],
                    }
                ]
            )
            
            return {
                "status": "completed",
                "content": message.content[0].text
            }

        except httpx.ConnectError as e:
            return {
                "status": "error",
                "error": "Connection error to Anthropic service",
                "detail": str(e)
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    async def _convert_pdf_page_to_image(self, page) -> bytes:
        """Convert a PDF page to an image using pdf2image"""
        try:
            # Create a temporary file for the page
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                # Write the page to a temporary PDF
                writer = PdfWriter()
                writer.add_page(page)
                writer.write(temp_pdf)
                temp_pdf.flush()
                
                # Convert the page to image
                images = convert_from_path(temp_pdf.name, dpi=200, fmt='jpeg')
                
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                
                return img_byte_arr.getvalue()
        finally:
            # Clean up temporary file
            if 'temp_pdf' in locals():
                os.unlink(temp_pdf.name)

    async def _process_single_page(self, b64_page: str, query: str, input_data_snapshot: str, page_number: int, total_pages: int) -> str:
        """Process a single page with Claude 3 Vision API"""
        try:
            # Convert base64 to image bytes
            image_bytes = base64.b64decode(b64_page)
            
            # Create PIL Image from bytes
            with Image.open(io.BytesIO(image_bytes)) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Save as JPEG to a new bytes buffer
                jpeg_buffer = io.BytesIO()
                img.save(jpeg_buffer, format='JPEG', quality=95)
                jpeg_buffer.seek(0)
                
                # Convert JPEG bytes to base64
                jpeg_b64 = base64.b64encode(jpeg_buffer.getvalue()).decode('utf-8')

            message = await asyncio.to_thread(
                self.client.messages.create,
                model=os.getenv("ANTHROPIC_MAIN_MODEL"),
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": jpeg_b64,
                                },
                            },
                            {
                                "type": "text",
                                "text": f"""Your job is to extract relevant information from this pdf image based on a user query and input data.
                                Extract only the relevant information from the image based on the query and data.
                                If formatting in the image provides information, indicate as much in your response 
                                (e.g. large text at the top of the image: title: [large text], 
                                tabular data: table: [tabular data], etc...).
                                Query and input data snapshot below in triple backticks.
                                ```Query: {query} 
                                Input Data Snapshot: 
                                {input_data_snapshot}
                                ```
                                """
                            }
                        ],
                    }
                ],
            )
            
            if not message or not message.content:
                logger.error("No response content from Anthropic")
                return None
                
            page_content = message.content[0].text
            if not page_content or len(page_content.strip()) == 0:
                logger.error("Empty content returned from Anthropic")
                return None
                
            print(f"""\n -------LLM called with query: {query} on page: {page_number} of {total_pages} ------- \n\n
            Input data snapshot:\n {input_data_snapshot}
            Page Content:\n {page_content}\n
            """)
            
            return page_content
            
        except Exception as e:
            logging.error(f"Error processing PDF page with vision: {str(e)}")
            return None


class LLMService:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.anthropic_client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self._operation_map = {
            "generate_text": {
                "openai": self._openai_generate_text,
                "anthropic": self._anthropic_generate_text
            },
            "process_image_with_vision": {
                "openai": self._openai_process_image_with_vision,
                "anthropic": self._anthropic_process_image_with_vision
            },
            "process_pdf_with_vision": {
                "openai": self._openai_process_pdf_with_vision,
                "anthropic": self._anthropic_process_pdf_with_vision
            },
            "gen_from_query": {
                "openai": self._openai_gen_from_query,
                "anthropic": self._anthropic_gen_from_query
            },
            "gen_from_error": {
                "openai": self._openai_gen_from_error,
                "anthropic": self._anthropic_gen_from_error
            },
            "gen_from_analysis": {
                "openai": self._openai_gen_from_analysis,
                "anthropic": self._anthropic_gen_from_analysis
            },
            "analyze_sandbox_result": {
                "openai": self._openai_analyze_sandbox_result,
                "anthropic": self._anthropic_analyze_sandbox_result
            },
            "sentiment_analysis": {
                "openai": self._openai_sentiment_analysis,
                "anthropic": self._anthropic_sentiment_analysis
            },
            "file_namer": {
                "openai": self._openai_file_namer,
                "anthropic": self._anthropic_file_namer
            },
            "gen_visualization": {
                "openai": self._openai_gen_visualization,
                "anthropic": self._anthropic_gen_visualization
            }
        }

        # Add system prompts as class attributes
        self._gen_from_query_prompt = gen_from_query_prompt
        self._gen_from_query_prompt_image = gen_from_query_prompt_image
        self._gen_from_error_prompt = gen_from_error_prompt
        self._gen_from_analysis_prompt = gen_from_analysis_prompt
        self._analyze_sandbox_prompt = analyze_sandbox_prompt
        self._sentiment_analysis_prompt = sentiment_analysis_prompt
        self._file_namer_prompt = file_namer_prompt
        self._gen_visualization_prompt = gen_visualization_prompt



    async def execute_with_fallback(self, operation: str, *args, **kwargs) -> Tuple[str, Any]:
        """Execute a function with fallback to another provider if the first fails"""
        
        try:
            # Try default provider first
            if os.getenv("DEFAULT_LLM_PROVIDER") == "anthropic":
                result = await self._execute_anthropic(operation, *args, **kwargs)
                return "anthropic", result
            else:  # default to openai
                result = await self._execute_openai(operation, *args, **kwargs)
                return "openai", result
            
        except Exception as e:
            # Handle other errors from default provider
            logging.warning(f"{os.getenv('DEFAULT_LLM_PROVIDER')} failed: {str(e)}. Falling back to alternate provider.")
            try:
                # Fallback to other provider
                if os.getenv("DEFAULT_LLM_PROVIDER") == "anthropic":
                    result = await self._execute_openai(operation, *args, **kwargs)
                    return "openai", result
                else:
                    result = await self._execute_anthropic(operation, *args, **kwargs)
                    return "anthropic", result
            except Exception as e2:
                raise ValueError(f"Both providers failed. Primary error: {str(e)}. Fallback error: {str(e2)}")

    async def _execute_openai(self, operation: str, *args, **kwargs):
        if operation not in self._operation_map:
            raise ValueError(f"Unknown operation: {operation}")
        return await self._operation_map[operation]["openai"](*args, **kwargs)

    async def _execute_anthropic(self, operation: str, *args, **kwargs):
        if operation not in self._operation_map:
            raise ValueError(f"Unknown operation: {operation}")
        return await self._operation_map[operation]["anthropic"](*args, **kwargs)

    async def _openai_generate_text(self, system_prompt: str, user_content: str) -> str:
        """Generate text using OpenAI with system and user prompts"""
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_MAIN_MODEL"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        )
        if not response.choices[0].message.content:
            raise ValueError("Empty response from API")
        return response.choices[0].message.content

    async def _anthropic_generate_text(self, system_prompt: str, user_content: str) -> str:
        """Generate text using Anthropic with system and user prompts"""
        response = self.anthropic_client.messages.create(
            model=os.getenv("ANTHROPIC_MAIN_MODEL"),
            max_tokens=5000,
            temperature=0,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        )
        if not response.content[0].text:
            raise ValueError("Empty response from API")
        return response.content[0].text

    async def _openai_process_image_with_vision(
        self, 
        image_path: str = None, 
        query: str = None, 
        input_data: List[FileDataInfo] = None,
        s3_key: str = None,
        use_s3: bool = False
    ) -> Dict[str, str]:
        """Process image using OpenAI's vision API. Raises an exception on connection errors 
           so that the fallback logic is triggered."""
        processor = OpenaiVisionProcessor(self.openai_client)
        result = await processor.process_image_with_vision(
            image_path=image_path,
            query=query,
            input_data=input_data,
            s3_key=s3_key,
            use_s3=use_s3
        )

        if result.get("status") == "error" and "connection error" in result.get("error", "").lower():
            import httpx
            raise httpx.ConnectError(f"OpenAI Connection Error: {result['error']}")

        return result

    async def _anthropic_process_image_with_vision(
        self,
        image_path: str = None,
        query: str = None,
        input_data: List[FileDataInfo] = None,
        s3_key: str = None,
        use_s3: bool = False
    ) -> Dict[str, str]:
        """Process image using Anthropic's vision API"""
        processor = AnthropicVisionProcessor(self.anthropic_client)
        return await processor.process_image_with_vision(
            image_path=image_path,
            query=query,
            input_data=input_data,
            s3_key=s3_key,
            use_s3=use_s3
        )

    async def _openai_process_pdf_with_vision(
        self, 
        pdf_path: str = None, 
        query: str = None, 
        input_data: List[FileDataInfo] = None, 
        page_range: Optional[tuple[int, int]] = None,
        s3_key: str = None,
        use_s3: bool = False,
        stream: BinaryIO = None
    ) -> Dict[str, str]:
        """Process PDF using OpenAI's vision API"""
        processor = OpenaiVisionProcessor(self.openai_client)
        # Ensure page_range values are integers if provided
        if page_range:
            page_range = (int(page_range[0]), int(page_range[1]))
        result = await processor.process_pdf_with_vision(
            pdf_path=pdf_path,
            s3_key=s3_key,
            query=query,
            input_data=input_data,
            page_range=page_range,
            use_s3=use_s3,
            stream=stream
        )

        # If the returned dict indicates an error related to connection, raise an actual exception
        if result.get("status") == "error" and "connection error" in result.get("error", "").lower():
            import httpx
            # Raise a ConnectError so execute_with_fallback will catch it
            raise httpx.ConnectError(f"OpenAI Connection Error: {result['error']}")

        return result

    async def _anthropic_process_pdf_with_vision(
        self, 
        pdf_path: str = None, 
        query: str = None, 
        input_data: List[FileDataInfo] = None, 
        page_range: Optional[tuple[int, int]] = None,
        s3_key: str = None,
        use_s3: bool = False,
        stream: BinaryIO = None
    ) -> Dict[str, str]:
        """Process PDF using Anthropic's vision API"""
        processor = AnthropicVisionProcessor(self.anthropic_client)
        # Ensure page_range values are integers if provided
        if page_range:
            page_range = (int(page_range[0]), int(page_range[1]))
        return await processor.process_pdf_with_vision(
            pdf_path=pdf_path,
            s3_key=s3_key,
            query=query,
            input_data=input_data,
            page_range=page_range,
            use_s3=use_s3,
            stream=stream
        )

    
    
    async def _openai_gen_from_query(
        self, 
        query: str, 
        data: List[FileDataInfo],
        batch_context: Optional[Dict[str, int]] = None,
        contains_image_or_like: bool = False
    ) -> str:
        data_description = self._build_data_description(data)
        
        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches.  This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f"Available Data:\n{data_description}\n\nQuery:\n{query}{batch_info}"
        
        response = await self._openai_generate_text(
            system_prompt=self._gen_from_query_prompt_image if contains_image_or_like else self._gen_from_query_prompt,
            user_content=user_content
        )
        print(f"\n -------  gen_from_query called with user content:  ------- \n {user_content}")
        return response

    async def _anthropic_gen_from_query(
        self, 
        query: str, 
        data: List[FileDataInfo],
        batch_context: Optional[Dict[str, int]] = None,
        contains_image_or_like: bool = False
    ) -> str:
        data_description = self._build_data_description(data)
        
        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches.  This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f"Available Data:\n{data_description}\n\nQuery:\n{query}{batch_info}"
        
        response = await self._anthropic_generate_text(
            system_prompt=self._gen_from_query_prompt_image if contains_image_or_like else self._gen_from_query_prompt,
            user_content=user_content
        )
        print(f"\n -------  gen_from_query called with user content:  ------- \n {user_content}")
        return response

    async def _openai_gen_from_error(self, result: SandboxResult, error_attempts: int, 
                                   data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Here is the original available data, user query, code, past errors, and new error
        - try not to repeat any of the past errors in your new solution:
        Available Data:\n{data_description}\n\n
        Original Query:\n{result.original_query}\n\n
        Code:\n{result.code}\n\n
        Past Errors:\n{past_errors}\n\n
        New Error:\n{result.error}
        """
        
        response = await self._openai_generate_text(
            system_prompt=self._gen_from_error_prompt,
            user_content=user_content
        )
        print(f"""Gen from error called, attempt: {error_attempts}, query: \n{result.original_query} 
              \ncode: \n{result.code} \nerror: \n{result.error}""")
        return response

    async def _anthropic_gen_from_error(self, result: SandboxResult, error_attempts: int,
                                      data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Here is the original available data, user query, code, past errors, and new error
                - try not to repeat any of the past errors in your new solution:
                Available Data:\n{data_description}\n\n
                Original Query:\n{result.original_query}\n\n
                Code:\n{result.code}\n\n
                Past Errors:\n{past_errors}\n\n
                New Error:\n{result.error}"""
        
        response = await self._anthropic_generate_text(
            system_prompt=self._gen_from_error_prompt,
            user_content=user_content
        )
        print(f"""Gen from error called, attempt: {error_attempts}, query: \n{result.original_query} 
              \ncode: \n{result.code} \nerror: \n{result.error}""")
        return response

    async def _openai_gen_from_analysis(self, result: SandboxResult, analysis_result: str,
                                      data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Original Query:\n{result.original_query}\n
        Available Data:\n{data_description}\n
        Code:\n{result.code}\n
        Analysis:\n{analysis_result}\n
        Past Errors:\n{past_errors}
        """
        
        return await self._openai_generate_text(
            system_prompt=self._gen_from_analysis_prompt,
            user_content=user_content
        )

    async def _anthropic_gen_from_analysis(self, result: SandboxResult, analysis_result: str,
                                         data: List[FileDataInfo], past_errors: List[str]) -> str:
        data_description = self._build_data_description(data)
        user_content = f"""Original Query:\n{result.original_query}\n
                Available Data:\n{data_description}\n
                Code:\n{result.code}\n
                Analysis:\n{analysis_result}\n
                Past Errors:\n{past_errors}
        """
        
        return await self._anthropic_generate_text(
            system_prompt=self._gen_from_analysis_prompt,
            user_content=user_content
        )

    async def _openai_analyze_sandbox_result(
        self, 
        result: SandboxResult, 
        old_data: List[FileDataInfo],
        new_data: FileDataInfo, 
        analyzer_context: str,
        batch_context: Optional[Dict[str, int]] = None
    ) -> str:
        old_data_snapshot = self._build_old_data_snapshot(old_data)
        if len(analyzer_context) < 10:
            analyzer_context = "No dataset diff information provided"

        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches. This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f""" 
    Here is the original user query, snapshots of input data, error free code, a snapshot of the result, and dataset diff information:
    Original Query:\n{result.original_query}\n
    Input Data Snapshots:\n{old_data_snapshot}\n
    Result Snapshot:\n{new_data.snapshot}\n
    {batch_info}\n
    Dataset Diff Information:\n{analyzer_context}\n
    """

        response = await self._openai_generate_text(
            system_prompt=self._analyze_sandbox_prompt,
            user_content=user_content
        )
        print(f"\n ------- analyze_sandbox_result called with user content: ------- \n {user_content}")
        return response

    async def _anthropic_analyze_sandbox_result(
        self, 
        result: SandboxResult, 
        old_data: List[FileDataInfo],
        new_data: FileDataInfo, 
        analyzer_context: str,
        batch_context: Optional[Dict[str, int]] = None
    ) -> str:
        old_data_snapshot = self._build_old_data_snapshot(old_data)
        if len(analyzer_context) < 10:
            analyzer_context = "No dataset diff information provided"

        # Add batch context to prompt if available
        batch_info = ""
        if batch_context:
            batch_info = f"\nUser files are being processed in batches. This is batch {batch_context['current']} of {batch_context['total']}"
        
        user_content = f""" 
    Here is the original user query, snapshots of input data, error free code, a snapshot of the result, and dataset diff information:
    Original Query:\n{result.original_query}\n
    Input Data Snapshots:\n{old_data_snapshot}\n
    Result Snapshot:\n{new_data.snapshot}\n
    {batch_info}\n
    Dataset Diff Information:\n{analyzer_context}\n
    """

        response = await self._anthropic_generate_text(
            system_prompt=self._analyze_sandbox_prompt,
            user_content=user_content
        )
        print(f"\n ------- analyze_sandbox_result called with user content: ------- \n {user_content}")
        return response

    async def _openai_sentiment_analysis(self, analysis_result: str) -> Tuple[bool, str]:
        response = await self._openai_generate_text(
            system_prompt=self._sentiment_analysis_prompt,
            user_content=f"Analysis:\n{analysis_result}"
        )
        try:
            result = json.loads(response)
            return result["is_positive"], analysis_result
        except json.JSONDecodeError:
            guess = "true" in response.lower()
            return guess, analysis_result

    async def _anthropic_sentiment_analysis(self, analysis_result: str) -> Tuple[bool, str]:
        response = await self._anthropic_generate_text(
            system_prompt=self._sentiment_analysis_prompt,
            user_content=f"Analysis:\n{analysis_result}"
        )
        try:
            result = json.loads(response)
            return result["is_positive"], analysis_result
        except json.JSONDecodeError:
            guess = "true" in response.lower()
            return guess, analysis_result

    async def _openai_file_namer(self, query: str, data: List[FileDataInfo]) -> str:
        """Generate a filename using OpenAI"""
        data_description = self._build_data_description(data)
        user_content = f"""Based on the query and data below, suggest a filename. 
        Avoid technical language (i.e. dataframe, list, etc.)
        Query: {query}
        Available Data: {data_description}"""
        
        response = self.openai_client.chat.completions.create(
            model=os.getenv("OPENAI_SMALL_MODEL"),
            messages=[
                {"role": "system", "content": self._file_namer_prompt},
                {"role": "user", "content": user_content}
            ]
        ).choices[0].message.content
        time.sleep(float(os.getenv("SLEEP_TIME")))

        
        return self._clean_filename(response)

    async def _anthropic_file_namer(self, query: str, data: List[FileDataInfo]) -> str:
        """Generate a filename using Anthropic"""
        data_description = self._build_data_description(data)
        user_content = f"""Based on the query and data below, suggest a filename. 
        Avoid technical language (i.e. dataframe, list, etc.)
        Query: {query}
        Available Data: {data_description}"""
        
        # Updated to use claude-3-5-haiku-20241022
        response = self.anthropic_client.messages.create(
            model=os.getenv("ANTHROPIC_SMALL_MODEL"),
            max_tokens=5000,
            temperature=0,
            system=self._file_namer_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": user_content
                        }
                    ]
                }
            ]
        ).content[0].text
        time.sleep(float(os.getenv("SLEEP_TIME")))
        return self._clean_filename(response)

    async def _openai_gen_visualization(
        self,
        data_snapshot: str,
        color_palette: str,
        custom_instructions: Optional[str],
        past_errors: List[str]
    ) -> str:
        """Generate visualization code using OpenAI."""
        print(f"Openai called with past errors: {past_errors}")
        return await self._openai_generate_text(
            system_prompt=self._gen_visualization_prompt,
            user_content=self._build_gen_vis_user_content(data_snapshot, color_palette, custom_instructions, past_errors)
        )

    async def _anthropic_gen_visualization(
        self,
        data_snapshot: str,
        color_palette: str,
        custom_instructions: Optional[str],
        past_errors: List[str]
    ) -> str:
        """Generate visualization code using Anthropic."""

        print(f"Anthropic called with past errors: {past_errors}")

        return await self._anthropic_generate_text(
            system_prompt=self._gen_visualization_prompt,
            user_content=self._build_gen_vis_user_content(data_snapshot, color_palette, custom_instructions, past_errors)
        )

    
    def _build_data_description(self, data: List[FileDataInfo]) -> str:
        if not data:
            return ""
        data_description = ""
        for idx, file_data in enumerate(data):
            var_name = f'data_{idx}' if idx > 0 else 'data'
            data_description += f"\nVariable Name: {var_name}\nData Type: {file_data.data_type}\nSnapshot:\n{file_data.snapshot}\n"
            if hasattr(file_data, 'original_file_name') and file_data.original_file_name:
                data_description += f"Original file name: {file_data.original_file_name}\n"
        return data_description

    def _build_old_data_snapshot(self, old_data: List[FileDataInfo]) -> str:
        old_data_snapshot = ""
        for data in old_data:
            if isinstance(data.snapshot, str):
                data_snapshot = data.snapshot[:MAX_SNAPSHOT_LENGTH] + "...cont'd"
            else:
                data_snapshot = data.snapshot
            old_data_snapshot += f"Original file name: {data.original_file_name}\nData type: {data.data_type}\nData Snapshot:\n{data_snapshot}\n\n"
        return old_data_snapshot

   
   
    def _build_gen_vis_user_content(self, data_snapshot: str, color_palette: str, custom_instructions: Optional[str], past_errors: List[str]) -> str:
        return f"""Data Snapshot:
        {data_snapshot}
        
        Incorporate these colors into your visualization: {color_palette}
        
        Custom Instructions: {custom_instructions if custom_instructions else 'None provided'}
        
        Generate visualization code following the requirements.
        
        Past Errors(please don't repeat any of these errors in your code): {past_errors}
        """

   
   
    def _clean_filename(self, filename: str) -> str:
        """Clean and standardize filename"""
        filename = filename.strip().lower()
        return "".join(c for c in filename if c.isalnum() or c in ['_', '-'])








