import pandas as pd
import json
from typing import Union, BinaryIO, Tuple, Optional, AsyncGenerator
from pathlib import Path
import docx
import requests
from PIL import Image
import io
from app.utils.s3_file_management import temp_file_manager
from app.utils.s3_file_actions import s3_file_actions, S3PDFStreamer
import fitz  # PyMuPDF
from tempfile import SpooledTemporaryFile
from typing import Dict, Tuple
from app.utils.data_processing import get_data_snapshot
from fastapi import UploadFile
from app.schemas import FileDataInfo, FileUploadMetadata, InputUrl
from typing import List
import logging
from app.utils.google_integration import GoogleIntegration
from app.utils.microsoft_integration import MicrosoftIntegration
from app.utils.llm_service import LLMService
from app.utils.auth import SupabaseClient
from fastapi import Request
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import aiohttp
import asyncio
import gc
from io import BytesIO
import tempfile

load_dotenv(override=True)

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def get_file_content(file: Union[BinaryIO, str, bytes, FileUploadMetadata, UploadFile]) -> Union[BinaryIO, AsyncGenerator[bytes, None]]:
    """Get file content from various sources including S3"""
    try:
        if isinstance(file, FileUploadMetadata):
            if file.s3_key:
                # Get content from S3
                return s3_file_actions.stream_download(file.s3_key)
            
        elif isinstance(file, UploadFile):
            # Handle UploadFile object
            return file.file
            
        elif isinstance(file, (str, bytes)):
            return io.BytesIO(file if isinstance(file, bytes) else file.encode())
            
        elif hasattr(file, 'read'):
            # Check if it's a file-like object with size information
            try:
                if hasattr(file, 'seek'):
                    file.seek(0)
                size = os.fstat(file.fileno()).st_size if hasattr(file, 'fileno') else None
                if size and size >= s3_file_actions.SMALL_FILE_THRESHOLD:
                    # For large files, stream in chunks
                    async def chunk_generator():
                        while True:
                            chunk = await asyncio.to_thread(file.read, s3_file_actions.CHUNK_SIZE)
                            if not chunk:
                                break
                            yield chunk
                    return chunk_generator()
            except (OSError, AttributeError):
                pass
            
            # Default to returning the file object directly
            return file
            
        raise ValueError("Unsupported file type or source")
    except Exception as e:
        logger.error(f"Error in get_file_content: {str(e)}")
        raise

async def pdf_classifier(file: Union[BinaryIO, str, bytes, FileUploadMetadata]) -> bool:
    """
    Classify if a PDF is image-like (scanned) or text-based.
    Returns True if the PDF is image-like (scanned), False if text-based.
    """
    doc = None
    try:
        # Handle S3 files using S3PDFStreamer
        if isinstance(file, FileUploadMetadata) and file.s3_key:
            try:
                # Use S3PDFStreamer for efficient page streaming
                pdf_streamer = S3PDFStreamer(s3_file_actions.s3_client, s3_file_actions.bucket, file.s3_key)
                
                # Only check first page
                if pdf_streamer.page_count > 0:
                    page_data = pdf_streamer.stream_page(1)
                    doc = fitz.open(stream=BytesIO(page_data), filetype="pdf")
                    if len(doc) > 0:
                        page = doc[0]  # More efficient than load_page(0)
                        page_text = page.get_text().strip()
                        return len(page_text) == 0
                return True
                
            except Exception as e:
                logger.error(f"Error in PDF classification using S3PDFStreamer: {str(e)}")
                return True  # Default to image-like on error
                
        else:
            # Handle direct file content
            try:
                # Get content as bytes
                if hasattr(file, 'read'):
                    if hasattr(file, 'seek'):
                        file.seek(0)
                    content = file.read()
                    if isinstance(content, str):
                        content = content.encode('utf-8')
                else:
                    content = file if isinstance(file, bytes) else file.encode('utf-8')
                
                # Use BytesIO for consistent handling
                doc = fitz.open(stream=BytesIO(content), filetype="pdf")
                if len(doc) > 0:
                    page = doc[0]  # More efficient than load_page(0)
                    page_text = page.get_text().strip()
                    return len(page_text) == 0
                return True
                
            except Exception as e:
                logger.error(f"Error in PDF classification for direct content: {str(e)}")
                return True  # Default to image-like on error
                
    finally:
        # Clean up resources
        if doc:
            try:
                doc.close()
            except Exception as e:
                logger.error(f"Error closing PDF document: {str(e)}")

class FilePreprocessor:
    """Handles preprocessing of various file types for data processing pipeline."""
    
    def __init__(self, num_images_processed: int = 0, supabase: SupabaseClient = None, user_id: str = None):
        """Initialize the FilePreprocessor with optional authentication"""
        self.num_images_processed = num_images_processed
        self.supabase = supabase
        self.user_id = user_id
        self.llm_service = LLMService()

    @staticmethod
    async def process_excel(file: Union[SpooledTemporaryFile, str, Path, BinaryIO, FileUploadMetadata, UploadFile]) -> pd.DataFrame:
        """Process Excel files (.xlsx) and convert to pandas DataFrame"""
        try:
            file_content = await get_file_content(file)
            # Handle BytesIO content from file.file
            if hasattr(file_content, 'read'):
                content = file_content.read()
                file_content.seek(0)
                if isinstance(content, bytes):
                    return pd.read_excel(io.BytesIO(content))
                return pd.read_excel(io.StringIO(content))
            
            return pd.read_excel(file_content)
        except Exception as e:
            raise ValueError(f"Error processing Excel file: {str(e)}")

    @staticmethod
    async def process_csv(file: Union[BinaryIO, str, FileUploadMetadata, UploadFile]) -> pd.DataFrame:
        """Process CSV files and convert to pandas DataFrame"""
        try:
            file_content = await get_file_content(file)
            
            # Handle BytesIO content from file.file
            if hasattr(file_content, 'read'):
                content = file_content.read()
                file_content.seek(0)
                if isinstance(content, bytes):
                    return pd.read_csv(io.BytesIO(content))
                return pd.read_csv(io.StringIO(content))
            
            return pd.read_csv(file_content)
        except Exception as e:
            raise ValueError(f"Error processing CSV file: {str(e)}")

    @staticmethod
    async def process_json(file: Union[BinaryIO, str, FileUploadMetadata, UploadFile]) -> str:
        """Process JSON files and convert to string"""
        try:
            file_content = await get_file_content(file)
            if isinstance(file_content, (str, bytes)):
                data = json.loads(file_content)
            else:
                data = json.load(file_content)
            return json.dumps(data)
        except Exception as e:
            raise ValueError(f"Error processing JSON file: {str(e)}")

    @staticmethod
    async def process_text(file: Union[BinaryIO, str, FileUploadMetadata, UploadFile]) -> str:
        """Process text files (.txt) and convert to string"""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        try:
            file_content = await get_file_content(file)
            content = file_content.read() if hasattr(file_content, 'read') else file_content
            
            if isinstance(content, str):
                return content
                
            for encoding in encodings:
                try:
                    return content.decode(encoding)
                except UnicodeDecodeError:
                    continue
            
            # If all encodings fail, try with error handling
            return content.decode('utf-8', errors='replace')
        except Exception as e:
            raise ValueError(f"Error processing text file: {str(e)}")

    @staticmethod
    async def process_docx(file: Union[BinaryIO, str, FileUploadMetadata, UploadFile]) -> str:
        """Process Word documents (.docx) and convert to string"""
        try:
            file_content = await get_file_content(file)
            
            # Create a fresh BytesIO object for docx.Document
            if hasattr(file_content, 'read'):
                content = file_content.read()
            else:
                content = file_content
                
            docx_file = io.BytesIO(content)
            doc = docx.Document(docx_file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Error processing DOCX file: {str(e)}")

    async def process_image(
        self, 
        file: Union[BinaryIO, str, FileUploadMetadata], 
        output_path: str = None,
        query: str = None,
        input_data: List[FileDataInfo] = None
    ) -> Tuple[str, str]:
        """Process image files and extract content using vision processing"""
        try:
            # Check limits before processing
            if self.supabase and self.user_id:
                await self.check_image_processing_limits(self.supabase, self.user_id)
            
            # Handle S3 files
            if isinstance(file, FileUploadMetadata) and file.s3_key:
                # Pass the S3 key directly to vision processor
                logger.info(f"Processing S3 file with key: {file.s3_key}")
                provider, vision_result = await self.llm_service.execute_with_fallback(
                    "process_image_with_vision",
                    s3_key=file.s3_key,
                    query=query,
                    input_data=input_data,
                    use_s3=True
                )

                if isinstance(vision_result, dict) and vision_result.get("status") == "error":
                    raise ValueError(f"Vision API error: {vision_result['error']}")
                
                self.num_images_processed += 1
                logger.info(f" -- Returning vision_result: {vision_result} and num_images_processed: {self.num_images_processed} --- ")
                return vision_result["content"] if isinstance(vision_result, dict) else vision_result, "s3"
            
            # For non-S3 files, use local processing
            file_content = await get_file_content(file)
            
            # Process the image
            if hasattr(file_content, 'read'):
                content = file_content.read()
                img = Image.open(io.BytesIO(content))
            else:
                # Handle string paths properly
                try:
                    if isinstance(file_content, str):
                        img = Image.open(Path(file_content))
                    elif isinstance(file_content, (bytes, bytearray)):
                        img = Image.open(io.BytesIO(file_content))
                    else:
                        img = Image.open(file_content)
                except Exception as e:
                    logger.error(f"Error opening image: {str(e)}")
                    raise ValueError(f"Unable to open image file: {str(e)}")
            
            # Convert PNG to JPEG if needed and save to S3
            img_buffer = io.BytesIO()
            if img.format == 'PNG' and img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(img_buffer, format='JPEG')
            img_buffer.seek(0)
            
            # Save to S3 using temp_file_manager
            if output_path is None:
                temp_dir = temp_file_manager.get_temp_dir()
                output_path = f"{temp_dir}temp_image.jpeg"
            
            s3_key = await temp_file_manager.save_temp_file(img_buffer, output_path.split('/')[-1])
            
            try:
                # Process with vision using LLM service
                provider, vision_result = await self.llm_service.execute_with_fallback(
                    "process_image_with_vision",
                    s3_key=s3_key,
                    query=query,
                    input_data=input_data,
                    use_s3=True
                )

                if isinstance(vision_result, dict) and vision_result.get("status") == "error":
                    raise ValueError(f"Vision API error: {vision_result['error']}")

                # Increment the image counter
                self.num_images_processed += 1

                result_content = vision_result["content"] if isinstance(vision_result, dict) else vision_result
                return result_content, output_path
            finally:
                # Mark the temporary S3 file for cleanup
                temp_file_manager.mark_for_cleanup(s3_key)
                await temp_file_manager.cleanup_marked()

        except Exception as e:
            raise ValueError(e)


    async def process_msft_excel_url(self, url, sheet_name: str = None, picker_token: str = None) -> pd.DataFrame:
        """
        Process Microsoft Excel URLs and convert to pandas DataFrame
        
        Args:
            input_url (InputUrl): InputUrl object containing the Microsoft Excel URL and metadata
            supabase (SupabaseClient): Supabase client for authentication
            user_id (str): User ID for authentication
            
        Returns:
            pd.DataFrame: DataFrame containing the sheet data
            
        Raises:
            ValueError: If URL is invalid or file cannot be accessed
        """
        
        supabase = self.supabase
        user_id = self.user_id
        
        try:
            if not supabase or not user_id:
                raise ValueError("Authentication required to access Microsoft Excel")

            # Initialize Microsoft integration W/O PICKER TOKEN
            msft_integration = MicrosoftIntegration(supabase, user_id)
            
            # Extract data using the sheet_name from input_url if provided
            return await msft_integration.extract_msft_excel_data(url, sheet_name)
                
        except Exception as e:
            raise ValueError(e)

    async def process_gsheet_url(self, url, sheet_name: str = None, picker_token: str = None) -> pd.DataFrame:
        """
        Process Google Sheets URLs and convert to pandas DataFrame
        
        Args:
            input_url (InputUrl): InputUrl object containing the Google Sheet URL and metadata
        
            
        Returns:
            pd.DataFrame: DataFrame containing the sheet data
            
        Raises:
            ValueError: If URL is invalid or file cannot be accessed
        """
        supabase = self.supabase
        user_id = self.user_id

        try:
            if not supabase or not user_id:
                raise ValueError("Authentication required to access Google Sheets")

            # Initialize Google integration
            g_integration = GoogleIntegration(supabase, user_id, picker_token)
            logger.info(f"g_integration obtained")
            # Pass sheet_name to extract_google_sheets_data
            return await g_integration.extract_google_sheets_data(url, sheet_name)
            logger.info(f"g_integration extracted data")
        except Exception as e:
            raise ValueError(e)

    async def process_pdf(
        self,
        file: Union[BinaryIO, str, FileUploadMetadata],
        output_path: str = None,
        query: str = None,
        input_data: List[FileDataInfo] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Tuple[str, str, bool]:
        """Process PDF files and extract content"""
        doc = None
        stream = None
        temp_path = None
        
        try:
            # For S3 files, process directly
            if isinstance(file, FileUploadMetadata) and file.s3_key:
                logger.info(f"Processing S3 PDF file: {file.s3_key}")
                try:
                    # Initialize S3PDFStreamer
                    streamer = S3PDFStreamer(s3_file_actions.s3_client, s3_file_actions.bucket, file.s3_key)
                    
                    # Get page range
                    start_page = page_range[0] if page_range else 0
                    end_page = min(page_range[1], streamer.page_count) if page_range else streamer.page_count
                    
                    text_content = []
                    total_text_length = 0
                    
                    # Process one page at a time
                    for page_num in range(start_page, end_page + 1):
                        try:
                            logger.info(f"Loading page {page_num}")
                            page_data = streamer.stream_page(page_num)
                            reader = PdfReader(BytesIO(page_data))
                            page_text = reader.pages[0].extract_text()
                            total_text_length += len(page_text.strip())
                            text_content.append(f"\n[Page {page_num} of {streamer.page_count} in user provided PDF]\n{page_text}")
                            logger.info(f"Successfully processed page {page_num}, text length: {len(page_text)}")
                        except Exception as page_e:
                            logger.error(f"Error processing page {page_num}: {str(page_e)}")
                            continue
                        finally:
                            gc.collect()

                    # If we got meaningful text content, return it
                    if total_text_length > 10:  # Same threshold as pdf_classifier
                        logger.info(f"Successfully extracted text content, total length: {total_text_length}")
                        return "\n".join(text_content), "text", False
                        
                except Exception as e:
                    logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
                
                # If we reach here, process with vision
                if self.supabase and self.user_id:
                    await self.check_image_processing_limits(self.supabase, self.user_id)
                
                # Pass the S3 key directly to vision processor
                provider, vision_result = await self.llm_service.execute_with_fallback(
                    "process_pdf_with_vision",
                    s3_key=file.s3_key,
                    query=query,
                    input_data=input_data,
                    page_range=page_range,
                    use_s3=True
                )

                if not vision_result:
                    raise ValueError("Failed to process PDF with vision")
                    
                if isinstance(vision_result, dict):
                    if vision_result.get("status") == "error":
                        raise ValueError(f"Vision API error: {vision_result.get('error')}")
                    content = vision_result.get("content")
                    if not content:
                        raise ValueError("No content returned from vision processing")
                else:
                    content = vision_result
                
                self.num_images_processed += (end_page - start_page)
                return content, "vision_extracted", True
            
            # For non-S3 files, use local processing
            file_content = await get_file_content(file)
            if hasattr(file_content, 'read'):
                content = file_content.read()
                stream = io.BytesIO(content)
            else:
                stream = io.BytesIO(file_content)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf.write(stream.getvalue())
                temp_pdf.flush()
                temp_path = temp_pdf.name
            
            try:
                # First try to process with Python PDF libraries
                doc = fitz.open(temp_path)
                page_count = len(doc)
                
                if page_range:
                    start_page = page_range[0] if page_range else 0
                    end_page = min(page_range[1], page_count) if page_range else page_count
                else:
                    start_page = 0
                    end_page = page_count

                text_content = []
                total_text_length = 0
                
                for page_num in range(start_page, end_page):
                    try:
                        page = doc.load_page(page_num)
                        page_text = page.get_text()
                        total_text_length += len(page_text.strip())
                        text_content.append(f"\n[Page {page_num + 1} of {page_count} in user provided PDF]\n{page_text}")
                    except Exception as e:
                        logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                        continue
                    finally:
                        # Clean up page resources immediately
                        if page:
                            page = None
                        gc.collect()

                # If we got meaningful text content, return it
                if total_text_length > 10:  # Same threshold as pdf_classifier
                    return "\n".join(text_content), "text", False

            except Exception as e:
                logging.error(f"Error reading PDF with Python libraries: {str(e)}")
            
            # If we reach here, process with vision
            if self.supabase and self.user_id:
                await self.check_image_processing_limits(self.supabase, self.user_id)
            
            try:
                # Reset stream position
                stream.seek(0)
                
                provider, vision_result = await self.llm_service.execute_with_fallback(
                    "process_pdf_with_vision",
                    pdf_path=None,  # Don't pass local path
                    query=query,
                    input_data=input_data,
                    page_range=page_range,
                    use_s3=False,
                    stream=stream  # Pass the stream directly
                )

                if not vision_result:
                    raise ValueError("Failed to process PDF with vision")
                    
                if isinstance(vision_result, dict):
                    if vision_result.get("status") == "error":
                        raise ValueError(f"Vision API error: {vision_result.get('error')}")
                    content = vision_result.get("content")
                    if not content:
                        raise ValueError("No content returned from vision processing")
                else:
                    content = vision_result
                
                self.num_images_processed += (end_page - start_page)
                return content, "vision_extracted", True

            finally:
                # Clean up resources
                if stream:
                    try:
                        stream.close()
                    except:
                        pass

        except Exception as e:
            logging.error(f"PDF processing error: {str(e)}", exc_info=True)
            raise ValueError(f"Error processing PDF: {str(e)}")
        
        finally:
            # Clean up resources in reverse order
            if stream:
                try:
                    stream.close()
                except:
                    pass
            if doc:
                try:
                    doc.close()
                except:
                    pass
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass

    #Class method to FilePreprocessor 
    async def preprocess_file(
        self, 
        file: Union[BinaryIO, str, FileUploadMetadata], 
        query: str, 
        file_type: str, 
        sheet_name: str = None, 
        picker_token: str = None,
        processed_data: List[FileDataInfo] = None,
        page_range: Optional[tuple[int, int]] = None,
    ) -> Union[str, pd.DataFrame]:
        """
        Preprocess file based on its type
        """
        processors = {
            'xlsx': self.process_excel,
            'csv': self.process_csv,
            'json': self.process_json,
            'txt': self.process_text,
            'docx': self.process_docx,
            'png': self.process_image,
            'jpg': self.process_image,
            'jpeg': self.process_image,
            'pdf': self.process_pdf,
            'gsheet': self.process_gsheet_url,
            'office_sheet': self.process_msft_excel_url  
        }
        processor = processors.get(file_type.lower())
        if not processor:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        input_data = processed_data
        # Handle async processors
        if file_type.lower() == 'pdf':
            return await processor(
                file, 
                output_path=None, 
                query=query, 
                input_data=input_data,
                page_range=page_range,
            )
        if file_type.lower() in ['png', 'jpg', 'jpeg']:
            return await processor(file, output_path=None, query=query, input_data=input_data)
        if file_type.lower() in ['gsheet', 'office_sheet']:
            url = ""
            url = file
            return await processor(url, sheet_name, picker_token)
        return await processor(file)

    async def check_image_processing_limits(self, supabase: SupabaseClient, user_id: str, num_pages: int = 1) -> None:
        """
        Check if user has exceeded image processing limits
        
        Args:
            supabase: Supabase client
            user_id: User ID
            num_pages: Number of pages/images to process (default 1)
            
        Raises:
            ValueError: If user has exceeded limits with specific message
        """
        # Get user profile and usage
        profile_response = supabase.table("user_profile").select("*").eq("id", user_id).execute()
        usage_response = supabase.table("user_usage").select("*").eq("user_id", user_id).execute()
        
        if not profile_response.data or not usage_response.data:
            raise ValueError("Could not fetch user data")
        
        profile = profile_response.data[0]
        usage = usage_response.data[0]
        
        # Get plan limits
        plan = profile.get("plan", "free")
        image_limit = {"free": 10, "pro": 200}.get(plan, 0)

        # Check image limit
        current_images = usage.get("images_processed_this_month", 0)
        if plan == "free" and current_images + num_pages > image_limit:
            raise ValueError("Image processing limit reached.")

        
        overage_this_month = usage.get("overage_this_month", 0)
        overage_hard_limit = usage.get("overage_hard_limit", 0)
        new_overage = overage_this_month + num_pages * 0.08
        greater_than_200 = current_images + num_pages > 200

        if plan == "pro" and new_overage >= overage_hard_limit and greater_than_200:
            raise ValueError("Monthly overage limit reached")



async def preprocess_files(
    files: List[UploadFile],
    files_metadata: List[FileUploadMetadata],
    input_urls: List[InputUrl],
    query: str,
    session_dir,
    supabase: SupabaseClient,
    user_id: str,
    num_images_processed: int = 0,
    page_range: Optional[tuple[int, int]] = None,
) -> Tuple[List[FileDataInfo], int]:
    """
    Preprocesses files and URLs, extracting their content for query processing.
    For batch processing, page_range specifies which pages to process.
    """
    
    preprocessor = FilePreprocessor(
        num_images_processed=num_images_processed,
        supabase=supabase,
        user_id=user_id
    )
    
    processed_data = []
    
    # Process web URLs if provided
    for input_url in input_urls:
        try:
            logging.info(f"Processing URL: {input_url.url}")
            if 'docs.google' in input_url.url:
                logging.info(f"Processing Google Sheet URL with sheet name: {input_url.sheet_name}")
                content = await preprocessor.preprocess_file(input_url.url, query, 'gsheet', input_url.sheet_name, input_url.picker_token)
                logging.info("Successfully processed Google Sheet") 
            elif 'onedrive.live' in input_url.url:
                logging.info(f"Processing OneDrive URL with sheet name: {input_url.sheet_name}")
                content = await preprocessor.preprocess_file(input_url.url, query, 'office_sheet', input_url.sheet_name, input_url.picker_token)
                logging.info("Successfully processed OneDrive file")
            else:
                logging.error(f"Unsupported URL format: {input_url.url}")
                continue
            
            data_type = "DataFrame" if isinstance(content, pd.DataFrame) else "text"
        
            processed_data.append(
                FileDataInfo(
                    content=content,
                    snapshot=get_data_snapshot(content, data_type),
                    data_type=data_type,
                    original_file_name=input_url.sheet_name if input_url.sheet_name else "url",
                    url=input_url.url
                )
            )
            logging.info(f"Successfully added processed data for URL: {input_url.url}")
            
        except Exception as e:
            error_msg = str(e)
            logging.error(f"Error processing URL {input_url.url}")
            raise ValueError(f"Error processing URL {input_url.url}")
    
    # Sort files_metadata to process CSV and XLSX files first
    if files_metadata:
        logger.info(f"files_metadata length: {len(files_metadata)}")
        priority_types = {
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'text/csv'
        }
        
        # Sort metadata while preserving original indices
        sorted_metadata = sorted(
            files_metadata,
            key=lambda x: (x.type not in priority_types, x.index)
        )
        logger.info(f"sorted_metadata: {sorted_metadata}")
        logger.info(f"files length: {len(files)}")
        logger.info(f"files_metadata length: {len(files_metadata)}")
        


    # Process uploaded files using metadata
    if files_metadata:
        logging.info("Beginning file processing")
        
        for metadata in sorted_metadata:
            try:
                if metadata.s3_key:
                    # Directly process S3 files using metadata
                    file = metadata
                    logging.info(f"Processing S3 file: {metadata.s3_key}")
                else:
                    # Use original index directly from metadata
                    if metadata.index >= len(files):
                        raise IndexError(f"Metadata index {metadata.index} exceeds files list length {len(files)}")
                    
                    file = files[metadata.index]
                    logging.info(f"Processing local file index {metadata.index}: {metadata.name}")

                # Map MIME types to FilePreprocessor types
                mime_to_processor = {
                    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
                    'text/csv': 'csv',
                    'application/json': 'json',
                    'text/plain': 'txt',
                    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
                    'image/png': 'png',
                    'image/jpeg': 'jpg',
                    'image/jpg': 'jpg',
                    'application/pdf': 'pdf'
                }
                
                file_type = mime_to_processor.get(metadata.type)
                if not file_type:
                    raise ValueError(f"Unsupported MIME type: {metadata.type}")
                is_image_like_pdf = False

                # Handle special cases for images and PDFs that need additional parameters
                kwargs = {}
                if file_type in ['png', 'jpg', 'jpeg']:
                    # Create S3 path for output
                    s3_prefix = temp_file_manager.get_temp_dir()
                    kwargs['output_path'] = f"{s3_prefix}{metadata.name}.jpeg"
                elif file_type == 'pdf':
                    kwargs['query'] = query

                # Process the file - handle S3 files differently
                if metadata.s3_key:
                    content = await preprocessor.preprocess_file(
                        metadata,  # Pass metadata instead of file_obj
                        query, 
                        file_type, 
                        sheet_name=None, 
                        processed_data=processed_data,
                        page_range=page_range,
                    )
                else:
                    with io.BytesIO(file.file.read()) as file_obj:
                        file_obj.seek(0)
                        content = await preprocessor.preprocess_file(
                            file_obj, 
                            query, 
                            file_type, 
                            sheet_name=None, 
                            processed_data=processed_data,
                            page_range=page_range,
                        )

                
                # Handle different return types
                if file_type == 'pdf':
                    content, data_type, is_image_like_pdf = content  # Unpack PDF processor return values
                    metadata_info = {"is_image_like_pdf": is_image_like_pdf}
                    
                else:
                    data_type = "DataFrame" if isinstance(content, pd.DataFrame) else "text"
                    metadata_info = {}

                logging.info(f"content: {content}, data_type: {data_type}, is_image_like_pdf: {is_image_like_pdf}")
                processed_data.append(
                    FileDataInfo(
                        content=content,
                        snapshot=get_data_snapshot(content, data_type, is_image_like_pdf),
                        data_type=data_type,
                        original_file_name=metadata.name,
                        metadata=metadata_info,
                        new_file_path=kwargs.get('output_path')  # For images
                    )
                )

            except Exception as e:
                error_msg = str(e)
                logging.error(f"Error processing file {metadata.name}: {error_msg}")
                raise ValueError(f"Error processing file {metadata.name}: {error_msg}")

    # logger.info(f" -- Returning processed_data: {processed_data} and num_images_processed: {preprocessor.num_images_processed} --- ")
    return processed_data, preprocessor.num_images_processed

async def determine_pdf_page_count(file: UploadFile) -> int:
    """Determines the page count of a PDF file"""
    if not file.content_type == "application/pdf":
        return 0
        
    try:
        # Read the file into memory
        contents = await file.read()
        pdf = PdfReader(io.BytesIO(contents))
        page_count = len(pdf.pages)
        
        # Reset file pointer for future reads
        await file.seek(0)
        
        return page_count
    except Exception as e:
        logger.error(f"Error determining PDF page count: {str(e)}")
        return 0

async def check_limits_pre_batch(supabase: SupabaseClient, user_id: str, total_pages: int = 1, job_id: str = None) -> None:
    """
    Check if user has exceeded image processing limits
    
    Args:
        supabase: Supabase client
        user_id: User ID
        num_pages: Number of pages/images to process (default 1)
        
    Raises:
        ValueError: If user has exceeded limits with specific message
    """
    # Get user profile and usage
    profile_response = supabase.table("user_profile").select("*").eq("id", user_id).execute()
    usage_response = supabase.table("user_usage").select("*").eq("user_id", user_id).execute()
    
    if not profile_response.data or not usage_response.data:
        raise ValueError("Could not fetch user data")
    
    profile = profile_response.data[0]
    usage = usage_response.data[0]
    
    # Get plan limits
    plan = profile.get("plan", "free")
    image_limit = {"free": int(os.getenv("FREE_USER_LIMIT")), "pro": int(os.getenv("PRO_USER_LIMIT"))}.get(plan, 0)
    
    # Check image limit
    current_images = usage.get("images_processed_this_month", 0)
    if current_images + total_pages > image_limit:
        if plan == "free":
            message = "This request would put you over your monthly image limit.  You can upgrade to pro in the account page to continue."
            supabase.table("jobs").update({
                "message": message
            }).eq("job_id", job_id).execute()
            supabase.table("jobs").update({
                "status": "error"
            }).eq("job_id", job_id).execute()
            raise ValueError(f"This request would put you over your monthly image limit.  You can upgrade to pro in the account page to continue.")
        # Check overage limit
        overage_this_month = usage.get("overage_this_month", 0)
        overage_hard_limit = usage.get("overage_hard_limit", 0)
        potential_overage = overage_this_month + total_pages * float(os.getenv("OVERAGE_PRICE"))
        print(f"potential_overage: {potential_overage}")
        print(f"overage_hard_limit: {overage_hard_limit}")
        if potential_overage >= overage_hard_limit:
            message = "This request would put you over your monthly overage limit.  You can increase your overage in your account page."
            supabase.table("jobs").update({
                "message": message
            }).eq("job_id", job_id).execute()
            supabase.table("jobs").update({
                "status": "error"
            }).eq("job_id", job_id).execute()
            raise ValueError(f"This request would put you over your monthly overage limit.  You can increase your overage in your account page.")

        