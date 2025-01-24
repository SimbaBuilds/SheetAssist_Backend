import os
from boto3 import client
from botocore.config import Config
import smart_open
import io
from typing import BinaryIO, Union, Optional
import asyncio
from botocore.exceptions import ClientError

class S3FileManager:
    def __init__(self):
        self.s3_client = client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION'),
            config=Config(signature_version='s3v4')
        )
        self.bucket = os.getenv('AWS_TEMP_BUCKET')

    async def stream_upload(self, file: BinaryIO, key: str) -> str:
        """Stream file directly to S3 without loading into memory"""
        try:
            await asyncio.to_thread(self.s3_client.upload_fileobj, file, self.bucket, key)
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            raise ValueError(f"Failed to upload to S3: {str(e)}")

    async def stream_download(self, key: str) -> BinaryIO:
        """Stream file from S3 without loading into memory"""
        try:
            file_obj = io.BytesIO()
            await asyncio.to_thread(self.s3_client.download_fileobj, self.bucket, key, file_obj)
            file_obj.seek(0)
            return file_obj
        except Exception as e:
            raise ValueError(f"Failed to download from S3: {str(e)}")

    def get_streaming_body(self, key: str) -> BinaryIO:
        """Get a streaming body for reading from S3"""
        try:
            return smart_open.open(f"s3://{self.bucket}/{key}", 'rb')
        except Exception as e:
            raise ValueError(f"Failed to get S3 stream: {str(e)}")

    def get_file_range(self, key: str, start: int, end: int) -> bytes:
        """Get specific byte range from S3 file"""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=key,
                Range=f'bytes={start}-{end}'
            )
            return response['Body'].read()
        except Exception as e:
            raise ValueError(f"Failed to get file range: {str(e)}")

    def get_presigned_url(self, key: str, expires_in: int = 3600, operation: str = 'get_object') -> str:
        """Generate a presigned URL for direct browser access"""
        try:
            return self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expires_in
            )
        except Exception as e:
            raise ValueError(f"Failed to generate presigned URL: {str(e)}")

    async def delete_file(self, key: str) -> bool:
        """Delete a file from S3"""
        try:
            await asyncio.to_thread(self.s3_client.delete_object, Bucket=self.bucket, Key=key)
            return True
        except Exception as e:
            raise ValueError(f"Failed to delete file: {str(e)}")

    def file_exists(self, key: str) -> bool:
        """Check if a file exists in S3"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
