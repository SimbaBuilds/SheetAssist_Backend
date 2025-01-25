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
import time
from app.dev_utils.memory_profiler import profile_memory

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_duration(start_time: float, operation: str) -> None:
    """Log the duration of an operation"""
    duration = time.time() - start_time
    logger.info(f"S3 operation '{operation}' completed in {duration:.2f} seconds")

class S3OperationError(Exception):
    """Custom exception for S3 operations"""
    pass

class S3FileActions:
    # Constants for file handling
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
    RETRY_ATTEMPTS = 3
    MIN_RETRY_WAIT = 1  # seconds
    MAX_RETRY_WAIT = 10  # seconds

    def __init__(self):
        logger.info("Initializing S3FileActions")
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
        logger.info(f"S3FileActions initialized with bucket: {self.bucket}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    @profile_memory
    async def stream_upload(self, file: BinaryIO, key: str) -> str:
        """Stream file directly to S3 with retry logic and proper error handling"""
        start_time = time.time()
        logger.info(f"Starting stream upload for key: {key}")
        try:
            if hasattr(file, 'seek'):
                file.seek(0)
                
            # Get file size if possible
            try:
                file_size = os.fstat(file.fileno()).st_size if hasattr(file, 'fileno') else None
                if file_size:
                    logger.debug(f"File size for {key}: {file_size} bytes")
            except (OSError, AttributeError):
                file_size = None
                logger.debug(f"Could not determine file size for {key}")

            # For small files, return None to indicate direct storage
            if file_size and file_size < self.SMALL_FILE_THRESHOLD:
                logger.debug(f"File {key} is below threshold, using direct storage")
                return None

            await asyncio.to_thread(self.s3_client.upload_fileobj, file, self.bucket, key)
            logger.info(f"Successfully uploaded file to S3: {key}")
            log_duration(start_time, "stream_upload")
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            logger.error(f"Failed to upload to S3: {str(e)}", exc_info=True)
            raise S3OperationError(f"Failed to upload to S3: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    @profile_memory
    async def stream_download(self, key: str) -> AsyncGenerator[bytes, None]:
        """Stream file from S3 in chunks"""
        start_time = time.time()
        logger.info(f"Starting stream download for key: {key}")
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key
            )
            stream = response['Body']
            
            chunk_count = 0
            total_bytes = 0
            while True:
                chunk = await asyncio.to_thread(stream.read, self.CHUNK_SIZE)
                if not chunk:
                    break
                chunk_count += 1
                total_bytes += len(chunk)
                yield chunk
            
            logger.info(f"Download completed for {key}. Total chunks: {chunk_count}, Total bytes: {total_bytes}")
            log_duration(start_time, "stream_download")
                
        except Exception as e:
            logger.error(f"Failed to download from S3: {str(e)}", exc_info=True)
            raise S3OperationError(f"Failed to download from S3: {str(e)}")
        finally:
            await asyncio.to_thread(stream.close)

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    @profile_memory
    async def get_streaming_body(self, key: str) -> BinaryIO:
        """Get a streaming body for reading from S3 with retry logic"""
        start_time = time.time()
        logger.info(f"Getting streaming body for key: {key}")
        try:
            stream = await asyncio.to_thread(
                smart_open.open,
                f"s3://{self.bucket}/{key}",
                'rb',
                transport_params={'client': self.s3_client}
            )
            log_duration(start_time, "get_streaming_body")
            return stream
        except Exception as e:
            logger.error(f"Failed to get S3 stream: {str(e)}", exc_info=True)
            raise S3OperationError(f"Failed to get S3 stream: {str(e)}")

    @retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=MIN_RETRY_WAIT, max=MAX_RETRY_WAIT),
        reraise=True
    )
    @profile_memory
    async def get_file_range(self, key: str, start: int, end: int) -> bytes:
        """Get specific byte range from S3 file with retry logic"""
        start_time = time.time()
        logger.info(f"Getting file range for key: {key}, range: {start}-{end}")
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key,
                Range=f'bytes={start}-{end}'
            )
            data = await asyncio.to_thread(response['Body'].read)
            logger.info(f"Successfully retrieved {len(data)} bytes from {key}")
            log_duration(start_time, "get_file_range")
            return data
        except Exception as e:
            logger.error(f"Failed to get file range: {str(e)}", exc_info=True)
            raise S3OperationError(f"Failed to get file range: {str(e)}")

# Global instance
s3_file_actions = S3FileActions()
