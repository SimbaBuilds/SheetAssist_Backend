from boto3 import client
from botocore.config import Config
import smart_open
import io
from typing import BinaryIO, Union

class S3FileManager:
    def __init__(self):
        self.s3_client = client(
            's3',
            config=Config(signature_version='s3v4')
        )
        self.bucket = "your-bucket-name"

    async def stream_upload(self, file: BinaryIO, key: str) -> str:
        """Stream file directly to S3 without loading into memory"""
        try:
            self.s3_client.upload_fileobj(file, self.bucket, key)
            return f"s3://{self.bucket}/{key}"
        except Exception as e:
            raise ValueError(f"Failed to upload to S3: {str(e)}")

    def get_streaming_body(self, key: str) -> BinaryIO:
        """Get a streaming body for reading from S3"""
        try:
            return smart_open.open(f"s3://{self.bucket}/{key}", 'rb')
        except Exception as e:
            raise ValueError(f"Failed to get S3 stream: {str(e)}")

    def get_presigned_url(self, key: str, expires_in: int = 3600) -> str:
        """Generate a presigned URL for direct browser upload"""
        try:
            return self.s3_client.generate_presigned_url(
                'put_object',
                Params={'Bucket': self.bucket, 'Key': key},
                ExpiresIn=expires_in
            )
        except Exception as e:
            raise ValueError(f"Failed to generate presigned URL: {str(e)}")
