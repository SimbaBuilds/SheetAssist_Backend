import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
from contextlib import asynccontextmanager
import asyncio
import boto3
from botocore.config import Config
import io
from typing import Union, BinaryIO
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class S3TempFileManager:
    def __init__(self, bucket_name: str, max_age_hours: int = 24, cleanup_interval_hours: int = 1):
        """Initialize the S3-based temporary file manager
        
        Args:
            bucket_name: S3 bucket for temp files
            max_age_hours: Maximum age of temp files before cleanup (in hours)
            cleanup_interval_hours: How often to run cleanup (in hours)
        """
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version='s3v4')
        )
        self.bucket = bucket_name
        self.max_age = timedelta(hours=max_age_hours)
        self._pending_cleanup = set()
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        self._cleanup_task = None

    def get_temp_dir(self) -> str:
        """Create a new temporary S3 prefix for the current session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_prefix = f"temp/session_{timestamp}/"
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
                            self.s3_client.delete_object(
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
        session_prefix: str = None
    ) -> str:
        """Save a temporary file to S3 and return its key
        
        Args:
            file_data: File data to save
            filename: Name for the saved file
            session_prefix: Optional specific session prefix to use
        """
        if session_prefix is None:
            session_prefix = self.get_temp_dir()

        # Generate unique filename to avoid collisions
        unique_id = str(uuid.uuid4())[:8]
        key = f"{session_prefix}{unique_id}_{filename}"

        try:
            if hasattr(file_data, 'read'):
                # For file-like objects, stream directly to S3
                self.s3_client.upload_fileobj(file_data, self.bucket, key)
            else:
                # For string or bytes content
                data = file_data.encode() if isinstance(file_data, str) else file_data
                self.s3_client.put_object(Bucket=self.bucket, Key=key, Body=data)
            
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
                self.s3_client.delete_object(Bucket=self.bucket, Key=key)
                logger.info(f"Cleaned up marked S3 object: {key}")
            except Exception as e:
                logger.error(f"Error cleaning up marked S3 object {key}: {str(e)}")

    async def get_file_stream(self, key: str) -> BinaryIO:
        """Get a file stream from S3"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            return response['Body']
        except Exception as e:
            raise ValueError(f"Failed to get S3 stream: {str(e)}")

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

# Global instance
temp_file_manager = S3TempFileManager(bucket_name=os.getenv('AWS_TEMP_BUCKET')) 