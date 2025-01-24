from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends   
from fastapi.responses import FileResponse
import os
from app.utils.s3_file_management import temp_file_manager
from typing import Annotated
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
import logging

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/download")
async def download_file(
    file_path: str,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> FileResponse:
    """
    Serve a processed file for download
    
    Args:
        file_path: Full path to the file to download
        background_tasks: FastAPI background tasks handler
        user_id: Authenticated user ID
        supabase: Supabase client instance
    """
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        # Determine the media type based on file extension
        filename = os.path.basename(file_path)
        extension = filename.split('.')[-1].lower()
        media_types = {
            'pdf': 'application/pdf',
            'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'txt': 'text/plain',
            'csv': 'text/csv',
            'png': 'image/png',
            'jpeg': 'image/jpeg',
            'jpg': 'image/jpeg'
        }
        
        media_type = media_types.get(extension, 'application/octet-stream')
        
        # Add cleanup task to run after file is sent
        background_tasks.add_task(temp_file_manager.cleanup_marked)
        
        logger.info(f"Serving file {filename} to user {user_id}")
        
        # Return the file as a response
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        logger.error(f"Error serving file for user {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


