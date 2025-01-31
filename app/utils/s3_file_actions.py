import os
import boto3
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
from pypdf import PdfReader
from io import BytesIO
import re
from pypdf import PdfWriter
from typing import Generator

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
        self.s3_client = boto3.client(
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
                    'checksum_validation': False,  
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

def find_xref_offset(s3_client, bucket: str, key: str) -> int:
    """Find the offset of the xref table by reading the last 1024 bytes."""
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
        Range='bytes=-1024'
    )
    end_data = response['Body'].read()
    
    # Find startxref position
    startxref_pos = end_data.rfind(b'startxref')
    if startxref_pos == -1:
        raise ValueError("Could not find xref table offset")
    
    # Extract the offset number
    offset_str = end_data[startxref_pos:].split(b'\n')[1].strip()
    return int(offset_str)

def parse_xref_table(s3_client, bucket: str, key: str, xref_offset: int) -> dict:
    """Parse the xref table to get object offsets."""
    # Get enough bytes to read the xref table
    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
        Range=f'bytes={xref_offset}-{xref_offset + 4096}'  # Get 4KB, should be enough for xref
    )
    xref_data = response['Body'].read()
    
    # Parse the xref entries
    object_offsets = {}
    xref_pattern = re.compile(rb'(\d{10}) (\d{5}) ([nf])')
    matches = xref_pattern.finditer(xref_data)
    
    for match in matches:
        offset, gen, used = match.groups()
        if used == b'n':  # Only store non-free objects
            object_offsets[len(object_offsets)] = int(offset)
    
    return object_offsets

def get_page_offsets(bucket: str, key: str) -> list[int]:
    """Get the actual byte offsets for each page in a PDF stored in S3."""
    s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION'),
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 3, 'mode': 'adaptive'},
                s3={
                    'payload_signing_enabled': True,
                    'use_accelerate_endpoint': False,
                    'addressing_style': 'path',
                    'checksum_validation': False,  
                    'use_dualstack_endpoint': False
                }
            )
        )
    
    # First, find the xref table offset
    xref_offset = find_xref_offset(s3, bucket, key)
    
    # Parse the xref table to get object offsets
    object_offsets = parse_xref_table(s3, bucket, key, xref_offset)
    
    # Get the root object to find pages
    root_offset = object_offsets[0]  # Usually the root is the first object
    response = s3.get_object(
        Bucket=bucket,
        Key=key,
        Range=f'bytes={root_offset}-{root_offset + 1024}'
    )
    root_data = response['Body'].read()
    
    # Extract page offsets from the root object
    page_offsets = []
    page_ref_pattern = re.compile(rb'/Kids\s*\[\s*(\d+\s+\d+\s+R[\s\d\sR]*)\]')
    match = page_ref_pattern.search(root_data)
    if match:
        page_refs = match.group(1).split(b'R')
        for ref in page_refs:
            if ref.strip():
                obj_num = int(ref.split()[0])
                if obj_num in object_offsets:
                    page_offsets.append(object_offsets[obj_num])
    
    return sorted(page_offsets)

class StreamingBuffer:
    """A streaming buffer that fetches data from S3 as needed."""
    
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    MAX_CACHE_SIZE = 10 * 1024 * 1024  # 10MB cache
    
    def __init__(self, s3: boto3.client, bucket: str, key: str, size: int):
        """Initialize the streaming buffer."""
        self.s3 = s3
        self.bucket = bucket
        self.key = key
        self.size = size
        self._pos = 0
        self._cache = {}  # chunk_start -> data
        self._cache_order = []  # LRU tracking
    
    def _get_chunk_start(self, pos: int) -> int:
        """Get the starting position of the chunk containing pos."""
        return (pos // self.CHUNK_SIZE) * self.CHUNK_SIZE
    
    def _fetch_chunk(self, chunk_start: int) -> bytes:
        """Fetch a chunk of data from S3."""
        if chunk_start in self._cache:
            # Move to end of LRU
            self._cache_order.remove(chunk_start)
            self._cache_order.append(chunk_start)
            return self._cache[chunk_start]
        
        chunk_end = min(chunk_start + self.CHUNK_SIZE, self.size)
        try:
            response = self.s3.get_object(
                Bucket=self.bucket,
                Key=self.key,
                Range=f'bytes={chunk_start}-{chunk_end-1}'
            )
            data = response['Body'].read()
            
            # Cache management
            while (len(self._cache_order) * self.CHUNK_SIZE) >= self.MAX_CACHE_SIZE:
                oldest = self._cache_order.pop(0)
                del self._cache[oldest]
            
            self._cache[chunk_start] = data
            self._cache_order.append(chunk_start)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching chunk at {chunk_start}: {str(e)}")
            raise
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """Seek to a position in the stream."""
        if whence == 0:  # SEEK_SET
            new_pos = offset
        elif whence == 1:  # SEEK_CUR
            new_pos = self._pos + offset
        elif whence == 2:  # SEEK_END
            new_pos = self.size + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        self._pos = max(0, min(new_pos, self.size))
        return self._pos
    
    def read(self, size: int = -1) -> bytes:
        """Read size bytes from the stream."""
        if size == -1:
            size = self.size - self._pos
        elif size < 0:
            raise ValueError("Size cannot be negative")

        if self._pos >= self.size:
            return b''

        # Calculate the chunks we need
        result = []
        bytes_remaining = min(size, self.size - self._pos)
        current_pos = self._pos
        
        while bytes_remaining > 0:
            chunk_start = self._get_chunk_start(current_pos)
            chunk_data = self._fetch_chunk(chunk_start)
            
            # Calculate offset within chunk
            chunk_offset = current_pos - chunk_start
            bytes_from_chunk = min(
                len(chunk_data) - chunk_offset,
                bytes_remaining
            )
            
            result.append(chunk_data[chunk_offset:chunk_offset + bytes_from_chunk])
            bytes_remaining -= bytes_from_chunk
            current_pos += bytes_from_chunk
        
        self._pos = current_pos
        return b''.join(result)
    
    def tell(self) -> int:
        """Return the current position in the stream."""
        return self._pos

class S3PDFStreamer:
    """Stream PDF files from S3 by page."""

    def __init__(self, s3_client, bucket: str, key: str):
        """Initialize the PDF streamer."""
        self.s3 = s3_client
        self.bucket = bucket
        self.key = key
        self.file_size = self._get_file_size()
        self.buffer = StreamingBuffer(s3_client, bucket, key, self.file_size)
        self._page_count = None  # Lazy load page count

    def _get_file_size(self) -> int:
        """Get the total file size from S3."""
        try:
            response = self.s3.head_object(Bucket=self.bucket, Key=self.key)
            return response['ContentLength']
        except Exception as e:
            logger.error(f"Error getting file size from S3: {str(e)}")
            raise

    @property
    def page_count(self) -> int:
        """Get the number of pages in the PDF (lazy loaded)."""
        if self._page_count is None:
            try:
                # Read the entire file for reliable page count
                self.buffer.seek(0)
                data = self.buffer.read()
                reader = PdfReader(BytesIO(data))
                self._page_count = len(reader.pages)
            except Exception as e:
                logger.error(f"Error getting page count: {str(e)}")
                # Default to 1 page if we can't determine count
                self._page_count = 1
        return self._page_count

    def stream_page(self, page_number: int) -> bytes:
        """Stream a specific page from the PDF."""

        try:
            # Read the entire file since we need it for reliable page extraction
            self.buffer.seek(0)
            data = self.buffer.read()
            reader = PdfReader(BytesIO(data))
            
            # Create a new PDF with just this page
            writer = PdfWriter()
            writer.add_page(reader.pages[page_number - 1])
            
            # Write to memory buffer
            output = BytesIO()
            writer.write(output)
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Error streaming page {page_number}: {str(e)}")
            raise

    def stream_full_document(self) -> Generator[bytes, None, None]:
        """Stream the entire PDF document page by page."""
        for page_num in range(1, self.page_count + 1):
            try:
                yield self.stream_page(page_num)
            except Exception as e:
                logger.error(f"Error streaming page {page_num}: {str(e)}")
                continue

# Global instance
s3_file_actions = S3FileActions()
