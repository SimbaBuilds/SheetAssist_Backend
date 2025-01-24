import os
from pathlib import Path
from datetime import datetime, timedelta
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3TempFileManager:
    def __init__(self, bucket_name: str = None, max_age_hours: int = 24, cleanup_interval_hours: int = 1):
        """Initialize the S3-based temporary file manager
        
        Args:
            bucket_name: S3 bucket for temp files (defaults to env var)
            max_age_hours: Maximum age of temp files before cleanup (in hours)
            cleanup_interval_hours: How often to run cleanup (in hours)
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION'),
            config=Config(signature_version='s3v4')
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
        try:
            cutoff_time = datetime.now() - self.max_age
            paginator = self.s3_client.get_paginator('list_objects_v2')
            
            async for page in paginator.paginate(Bucket=self.bucket, Prefix='temp/'):
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    if obj['LastModified'] < cutoff_time:
                        try:
                            await asyncio.to_thread(
                                self.s3_client.delete_object,
                                Bucket=self.bucket,
                                Key=obj['Key']
                            )
                            logger.info(f"Cleaned up old S3 file: {obj['Key']}")
                        except Exception as e:
                            logger.error(f"Error cleaning up {obj['Key']}: {str(e)}")
                            
        except Exception as e:
            logger.error(f"Error during S3 cleanup: {str(e)}")

    async def save_temp_file(
        self, 
        file_data: Union[BinaryIO, bytes, str], 
        filename: str, 
        session_prefix: str = None,
        metadata: dict = None
    ) -> str:
        """Save a temporary file to S3 and return its key
        
        Args:
            file_data: File data to save
            filename: Name for the saved file
            session_prefix: Optional specific session prefix to use
            metadata: Optional metadata to store with the file
        """
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

            if hasattr(file_data, 'read'):
                # For file-like objects, stream directly to S3
                await asyncio.to_thread(
                    self.s3_client.upload_fileobj,
                    file_data,
                    self.bucket,
                    key,
                    ExtraArgs=extra_args
                )
            else:
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
            
            return key
        except Exception as e:
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

    async def get_file_stream(self, key: str) -> BinaryIO:
        """Get a file stream from S3"""
        try:
            response = await asyncio.to_thread(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key
            )
            return response['Body']
        except Exception as e:
            raise ValueError(f"Failed to get S3 stream: {str(e)}")

    async def get_file_metadata(self, key: str) -> Optional[dict]:
        """Get metadata for a file if it exists"""
        try:
            if key in self._session_metadata:
                return self._session_metadata[key]
            
            response = await asyncio.to_thread(
                self.s3_client.head_object,
                Bucket=self.bucket,
                Key=key
            )
            return response.get('Metadata')
        except ClientError:
            return None

    async def list_session_files(self, session_prefix: str) -> List[str]:
        """List all files in a session directory"""
        try:
            files = []
            paginator = self.s3_client.get_paginator('list_objects_v2')
            async for page in paginator.paginate(Bucket=self.bucket, Prefix=session_prefix):
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