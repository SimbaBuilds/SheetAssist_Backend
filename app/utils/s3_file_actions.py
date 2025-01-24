import os
from boto3 import client
from botocore.config import Config
import smart_open
import io
from typing import BinaryIO, Union, Optional, AsyncGenerator, Dict
import asyncio
from botocore.exceptions import ClientError
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

class S3OperationError(Exception):
    """Custom exception for S3 operations"""
    pass

class S3FileManager:
    # Constants for file handling
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
    SMALL_FILE_THRESHOLD = 100 * 1024  # 100KB threshold for direct storage
    RETRY_ATTEMPTS = 3
    MIN_RETRY_WAIT = 1  # seconds
    MAX_RETRY_WAIT = 10  # seconds

    def __init__(self):
        self.s3_client = client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION'),
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': self.RETRY_ATTEMPTS, 'mode': 'adaptive'}
            )
        )
        self.bucket = os.getenv('AWS_TEMP_BUCKET')

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def stream_upload(self, file: BinaryIO, key: str) -> str:
        """Stream file directly to S3 with retry logic and proper error handling"""
        try:
            if hasattr(file, 'seek'):
                file.seek(0)
                
            # Get file size if possible
            try:
                file_size = os.fstat(file.fileno()).st_size if hasattr(file, 'fileno') else None
            except (OSError, AttributeError):
                file_size = None

            # For small files, return None to indicate direct storage
            if file_size and file_size < self.SMALL_FILE_THRESHOLD:
                return None

            await asyncio.to_thread(self.s3_client.upload_fileobj, file, self.bucket, key)
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}")
            raise S3OperationError(f"Failed to upload to S3: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def stream_download(self, key: str) -> AsyncGenerator[bytes, None]:
        """Stream file from S3 in chunks"""
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key
            )
            stream = response['Body']
            
            while True:
                chunk = await asyncio.to_thread(stream.read, self.CHUNK_SIZE)
                if not chunk:
                    break
                yield chunk
                
        except Exception as e:
            logger.error(f"Failed to download from S3: {str(e)}")
            raise S3OperationError(f"Failed to download from S3: {str(e)}")
        finally:
            await asyncio.to_thread(stream.close)

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    def get_streaming_body(self, key: str) -> BinaryIO:
        """Get a streaming body for reading from S3 with retry logic"""
        try:
            return smart_open.open(f"s3://{self.bucket}/{key}", 'rb', transport_params={'client': self.s3_client})
        except Exception as e:
            logger.error(f"Failed to get S3 stream: {str(e)}")
            raise S3OperationError(f"Failed to get S3 stream: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def get_file_range(self, key: str, start: int, end: int) -> bytes:
        """Get specific byte range from S3 file with retry logic"""
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key,
                Range=f'bytes={start}-{end}'
            )
            return await asyncio.to_thread(response['Body'].read)
        except Exception as e:
            logger.error(f"Failed to get file range: {str(e)}")
            raise S3OperationError(f"Failed to get file range: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL with retry logic"""
        try:
            return self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expires_in
            )
        except Exception as e:
            logger.error(f"Failed to generate presigned URL: {str(e)}")
            raise S3OperationError(f"Failed to generate presigned URL: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def delete_file(self, key: str) -> bool:
        """Delete a file from S3 with retry logic"""
        try:
            await asyncio.to_thread(self.s3_client.delete_object, Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            raise S3OperationError(f"Failed to delete file: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    def file_exists(self, key: str) -> bool:
        """Check if a file exists in S3 with retry logic"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
        except Exception as e:
            logger.error(f"Error checking file existence: {str(e)}")
            raise S3OperationError(f"Error checking file existence: {str(e)}")

    async def get_file_size(self, key: str) -> int:
        """Get the size of a file in S3"""
        try:
            response = await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=self.bucket,
                Key=key
            )
            return response['ContentLength']
        except Exception as e:
            logger.error(f"Failed to get file size: {str(e)}")
            raise S3OperationError(f"Failed to get file size: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    async def stream_pdf_by_page(self, key: str, start_page: int = None, end_page: int = None) -> AsyncGenerator[Dict[str, Union[bytes, int]], None]:
        """
        Stream PDF from S3 page by page, ensuring we don't split pages.
        Returns a generator that yields dictionaries containing page data and metadata.
        """
        try:
            # First get the PDF metadata to know total pages
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key
            )
            
            # Create a temporary file to store the PDF
            temp_file = io.BytesIO()
            stream = response['Body']
            
            # Stream the file in chunks to the temporary buffer
            while True:
                chunk = await asyncio.to_thread(stream.read, self.CHUNK_SIZE)
                if not chunk:
                    break
                temp_file.write(chunk)
            
            temp_file.seek(0)
            
            # Open with PyMuPDF
            doc = fitz.open(stream=temp_file)
            total_pages = len(doc)
            
            # Determine page range
            start = start_page if start_page is not None else 0
            end = min(end_page, total_pages) if end_page is not None else total_pages
            
            # Yield each page
            for page_num in range(start, end):
                page = doc[page_num]
                
                # Get page as bytes (PNG format for image-like PDFs)
                pix = page.get_pixmap()
                img_bytes = pix.tobytes()
                
                yield {
                    'page_number': page_num + 1,
                    'total_pages': total_pages,
                    'content': img_bytes,
                    'width': pix.width,
                    'height': pix.height
                }
            
            doc.close()
            temp_file.close()
                
        except Exception as e:
            logger.error(f"Failed to stream PDF by page: {str(e)}")
            raise S3OperationError(f"Failed to stream PDF by page: {str(e)}")
        finally:
            await asyncio.to_thread(stream.close)
