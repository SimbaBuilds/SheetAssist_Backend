from fastapi import APIRouter, HTTPException, BackgroundTasks   
from fastapi.responses import FileResponse
import os
from app.utils.file_management import temp_file_manager


router = APIRouter()


@router.get("/download")
async def download_file(
    file_path: str,
    background_tasks: BackgroundTasks
) -> FileResponse:
    """
    Serve a processed file for download
    
    Args:
        file_path: Full path to the file to download
        background_tasks: FastAPI background tasks handler
    """
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
            'csv': 'text/csv'
        }
        
        media_type = media_types.get(extension, 'application/octet-stream')
        
        # Add cleanup task to run after file is sent
        background_tasks.add_task(temp_file_manager.cleanup_marked)
        
        # Return the file as a response
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")


