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
    CHUNK_SIZE = 3 * 1024 * 1024  # 3MB chunks for streaming
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
                retries={'max_attempts': self.RETRY_ATTEMPTS, 'mode': 'adaptive'},
                s3={
                    'payload_signing_enabled': True,
                    'use_accelerate_endpoint': False,
                    'addressing_style': 'path',
                    'checksum_validation': True,  # Enable checksum validation
                    'use_dualstack_endpoint': False
                }
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

    class SeekableStreamWrapper:
        """Makes a non-seekable stream appear seekable by buffering only what's needed."""
        
        def __init__(self, stream):
            self.stream = stream
            self._buffer = io.BytesIO()
            self._position = 0
            self._eof = False
            
        def tell(self):
            return self._position
            
        def seek(self, offset, whence=io.SEEK_SET):
            if whence == io.SEEK_SET:
                position = offset
            elif whence == io.SEEK_CUR:
                position = self._position + offset
            else:  # io.SEEK_END - we need to read the whole stream
                if offset != 0:
                    raise io.UnsupportedOperation("can't do nonzero end-relative seeks")
                # Read the rest of the stream
                while not self._eof:
                    chunk = self.stream.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        self._eof = True
                        break
                    self._buffer.write(chunk)
                position = self._buffer.getbuffer().nbytes + offset
                
            # If we need to read more data
            if position > self._buffer.getbuffer().nbytes and not self._eof:
                to_read = position - self._buffer.getbuffer().nbytes
                while to_read > 0 and not self._eof:
                    chunk = self.stream.read(min(1024 * 1024, to_read))
                    if not chunk:
                        self._eof = True
                        break
                    self._buffer.write(chunk)
                    to_read -= len(chunk)
                    
            self._position = min(position, self._buffer.getbuffer().nbytes)
            self._buffer.seek(self._position)
            return self._position
            
        def read(self, size=-1):
            if size == -1:
                # Read the rest of the buffer
                data = self._buffer.read()
                # Then read the rest of the stream
                while not self._eof:
                    chunk = self.stream.read(1024 * 1024)  # 1MB chunks
                    if not chunk:
                        self._eof = True
                        break
                    self._buffer.write(chunk)
                    data += chunk
                self._position = self._buffer.tell()
                return data
                
            # First try reading from buffer
            data = self._buffer.read(size)
            read_size = len(data)
            
            # Need more data from stream
            if read_size < size and not self._eof:
                remaining = size - read_size
                chunk = self.stream.read(remaining)
                if not chunk:
                    self._eof = True
                else:
                    self._buffer.write(chunk)
                    data += chunk
                
            self._position = self._buffer.tell()
            return data
            
        def seekable(self):
            return True
            
        def readable(self):
            return True
            
        def close(self):
            self.stream.close()
            self._buffer.close()

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
            # Get raw stream from S3
            logger.info("Calling S3 get_object")
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key
            )
            
            # Get the raw stream
            stream = response['Body']
            
            # Create a seekable wrapper around the stream
            seekable_stream = self.SeekableStreamWrapper(stream)
            logger.info("Created seekable stream wrapper")
            
            log_duration(start_time, "get_streaming_body")
            return seekable_stream
            
        except Exception as e:
            logger.error(f"Failed to get S3 stream: {str(e)}", exc_info=True)
            raise S3OperationError(f"Failed to get S3 stream: {str(e)}")
        finally:
            if 'stream' in locals() and not isinstance(stream, self.SeekableStreamWrapper):
                await asyncio.to_thread(stream.close)

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
