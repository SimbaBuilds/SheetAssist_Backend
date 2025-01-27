import os
from pathlib import Path
from datetime import datetime, timedelta, UTC
import logging
from contextlib import asynccontextmanager
import asyncio
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import io
from typing import Union, BinaryIO, Optional, List
import uuid
import json
import time

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

class S3TempFileManager:
    def __init__(self, bucket_name: str = None, max_age_hours: int = 24, cleanup_interval_hours: int = 1):
        """Initialize the S3-based temporary file manager
        
        Args:
            bucket_name: S3 bucket for temp files (defaults to env var)
            max_age_hours: Maximum age of temp files before cleanup (in hours)
            cleanup_interval_hours: How often to run cleanup (in hours)
        """
        logger.info("Initializing S3TempFileManager")
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION'),
            config=Config(
                signature_version='s3v4',
                s3={
                    'payload_signing_enabled': True,
                    'use_accelerate_endpoint': False,
                    'addressing_style': 'path',
                    'checksum_validation': False,  # Disable checksum validation
                    'use_dualstack_endpoint': False
                }
            )
        )
        self.bucket = bucket_name or os.getenv('AWS_TEMP_BUCKET')
        if not self.bucket:
            raise ValueError("No S3 bucket specified and AWS_TEMP_BUCKET not set")
        self.max_age = timedelta(hours=max_age_hours)
        self._pending_cleanup = set()
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        self._cleanup_task = None
        self._session_metadata = {}

    def get_temp_dir(self, prefix: str = None) -> str:
        """Create a new temporary S3 prefix for the current session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = str(uuid.uuid4())[:8]
        session_prefix = f"temp/{prefix + '_' if prefix else ''}{session_id}_{timestamp}/"
        return session_prefix

    async def cleanup_old_files(self):
        """Remove temporary files older than max_age from S3"""
        start_time = time.time()
        logger.info("Starting cleanup of old S3 files")
        try:
            cutoff_time = datetime.now(UTC) - self.max_age
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            # Get all pages synchronously within a thread
            pages = await asyncio.to_thread(
                lambda: list(paginator.paginate(Bucket=self.bucket, Prefix='temp/'))
            )
            
            cleaned_count = 0
            for page in pages:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    # S3 returns timezone-aware datetime, so we can compare directly
                    if obj['LastModified'] < cutoff_time:
                        try:
                            await asyncio.to_thread(
                                self.s3_client.delete_object,
                                Bucket=self.bucket,
                                Key=obj['Key']
                            )
                            cleaned_count += 1
                            logger.debug(f"Cleaned up old S3 file: {obj['Key']}")
                        except Exception as e:
                            logger.error(f"Error cleaning up {obj['Key']}: {str(e)}")
            
            logger.info(f"Cleanup completed. Removed {cleaned_count} old files")
            log_duration(start_time, "cleanup_old_files")
                            
        except Exception as e:
            logger.error(f"Error during S3 cleanup: {str(e)}")

    async def save_temp_file(
        self, 
        file_data: Union[BinaryIO, bytes, str], 
        filename: str, 
        session_prefix: str = None,
        metadata: dict = None
    ) -> str:
        """Save a temporary file to S3 and return its key"""
        start_time = time.time()
        logger.info(f"Saving temporary file: {filename}")
        
        if session_prefix is None:
            session_prefix = self.get_temp_dir()

        # Generate unique filename to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        key = f"{session_prefix}{unique_id}_{filename}"
        
        try:
            extra_args = {}
            if metadata:
                extra_args['Metadata'] = {
                    k: str(v) for k, v in metadata.items()
                }
                logger.debug(f"Adding metadata to file {key}: {metadata}")

            if hasattr(file_data, 'read'):
                logger.debug(f"Streaming file-like object to S3: {key}")
                # For file-like objects, stream directly to S3
                await asyncio.to_thread(
                    self.s3_client.upload_fileobj,
                    file_data,
                    self.bucket,
                    key,
                    ExtraArgs=extra_args
                )
            else:
                logger.debug(f"Uploading string/bytes content to S3: {key}")
                # For string or bytes content
                data = file_data.encode() if isinstance(file_data, str) else file_data
                await asyncio.to_thread(
                    self.s3_client.put_object,
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    **extra_args
                )
            
            if metadata:
                self._session_metadata[key] = metadata
                
            logger.info(f"Successfully saved file to S3: {key}")
            log_duration(start_time, "save_temp_file")
            return key
            
        except Exception as e:
            logger.error(f"Failed to save file to S3: {str(e)}", exc_info=True)
            raise ValueError(f"Failed to save file to S3: {str(e)}")

    def mark_for_cleanup(self, *keys: str) -> None:
        """Mark S3 objects for cleanup"""
        for key in keys:
            self._pending_cleanup.add(key)

    async def cleanup_marked(self) -> None:
        """Clean up all marked S3 objects"""
        while self._pending_cleanup:
            key = self._pending_cleanup.pop()
            try:
                await asyncio.to_thread(
                    self.s3_client.delete_object,
                    Bucket=self.bucket,
                    Key=key
                )
                logger.info(f"Cleaned up marked S3 object: {key}")
                self._session_metadata.pop(key, None)
            except Exception as e:
                logger.error(f"Error cleaning up marked S3 object {key}: {str(e)}")

    async def get_file_metadata(self, key: str) -> Optional[dict]:
        """Get metadata for a file if it exists"""
        start_time = time.time()
        logger.info(f"Getting metadata for: {key}")
        try:
            if key in self._session_metadata:
                logger.info(f"Found metadata in cache for: {key}")
                return self._session_metadata[key]
            
            response = await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=self.bucket,
                Key=key
            )
            logger.info(f"Response: {response}")
            log_duration(start_time, "get_file_metadata")
            
            # Return the full response if Metadata is empty
            # This ensures we still get ContentLength and other important attributes
            return response if not response.get('Metadata') else response.get('Metadata')
            
        except ClientError as e:
            logger.warning(f"No metadata found for {key}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error getting metadata for {key}: {str(e)}", exc_info=True)
            return None

    async def list_session_files(self, session_prefix: str) -> List[str]:
        """List all files in a session directory"""
        try:
            files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            # Get all pages synchronously within a thread
            pages = await asyncio.to_thread(
                lambda: list(paginator.paginate(Bucket=self.bucket, Prefix=session_prefix))
            )
            
            for page in pages:
                if 'Contents' in page:
                    files.extend(obj['Key'] for obj in page['Contents'])
            return files
        except Exception as e:
            raise ValueError(f"Failed to list session files: {str(e)}")

    async def start_periodic_cleanup(self):
        """Start periodic cleanup task"""
        async def cleanup_loop():
            while True:
                await self.cleanup_old_files()
                await asyncio.sleep(self.cleanup_interval.total_seconds())

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_periodic_cleanup(self):
        """Stop periodic cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    @asynccontextmanager
    async def session_context(self, prefix: str = None):
        """Context manager for handling a temporary file session"""
        session_prefix = self.get_temp_dir(prefix)
        try:
            yield session_prefix
        finally:
            # Clean up all files in the session
            files = await self.list_session_files(session_prefix)
            self.mark_for_cleanup(*files)
            await self.cleanup_marked()

# Global instance
temp_file_manager = S3TempFileManager() 