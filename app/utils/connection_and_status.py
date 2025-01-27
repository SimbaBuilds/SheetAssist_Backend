from fastapi import Request
import logging
import asyncio
from app.schemas import QueryResponse, FileInfo, TruncatedSandboxResult
import os

logger = logging.getLogger(__name__)

# Add client disconnection check
async def check_client_connection(request: Request) -> bool:
    """Enhanced connection check with retry logic"""
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    for attempt in range(MAX_RETRIES):
        try:
            if await request.is_disconnected():
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                logger.warning("Client disconnected after retries")
                return False
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            logger.error(f"Connection check error: {e}")
            return False
    
    return True



def construct_status_response(job: dict) -> QueryResponse:
    """Helper function to construct status response"""

    # Get the current status and data for the job
    current_status = job.get("status", "unknown")
    page_chunks = job.get('page_chunks', [])
    current_chunk = int(job.get('current_chunk', 0))
    message = job.get("message")
    file_id = page_chunks[current_chunk]['file_id'] # the file name
    output_preferences = job.get('output_preferences')

    if output_preferences.get('doc_name'):
        doc_name = output_preferences.get('doc_name')
        sheet_name = output_preferences.get('sheet_name')
    else:
        doc_name = None
        sheet_name = None
    
    # Handle cases
    if current_status == "completed":
        if output_preferences['type'] == 'online':
            new_message = message + f"Processing complete."

            return QueryResponse(
                status=current_status,
                message=new_message,
                num_images_processed=job["total_images_processed"],
            )
        else:  # download type
            new_message = message + f"Processing complete. Your file should download automatically"
            return QueryResponse(
                status=current_status,
                message=new_message,
                files=[FileInfo(
                    file_path=job["result_file_path"],
                    media_type=job["result_media_type"],
                    filename=os.path.basename(job["result_file_path"]),
                    download_url=f"/download?file_path={job['result_file_path']}"
                )],
                num_images_processed=job["total_images_processed"],
            )
    
    elif current_status == "error":
        return QueryResponse(
            status="error",
            message=job["error_message"],
            num_images_processed=job["total_images_processed"],
        )
    
    elif current_status == "created":
        return QueryResponse(
            status="created",
            message=f"Processing pages {page_chunks[current_chunk]['page_range'][0] + 1} to {page_chunks[current_chunk]['page_range'][1]}",
            num_images_processed=job["total_images_processed"],
        )
    
    else:  # processing        
        start_page = int(page_chunks[current_chunk]['page_range'][0]) + 1
        end_page = int(page_chunks[current_chunk]['page_range'][1])
        current_chunk = int(job.get('current_chunk', 0))
        chunk_status = job.get("chunk_status", [])
        chunk_success = "Success" in chunk_status[current_chunk] if current_chunk < len(chunk_status) else False
        logger.info(f"CHUNK STATUS: {chunk_status[current_chunk]}, Current chunk: {current_chunk}, Chunk SUCCESS: {chunk_success}")
        if output_preferences['type'] == 'online':
            if output_preferences['modify_existing']:
                new_message = message + f"Page {max(0, start_page)} to {end_page} from file {file_id} processed and appended to {doc_name} - {sheet_name}.\n" if chunk_success else f"FAILED to successfully process page {max(0, start_page)} to {end_page} from file {file_id}.  Please inspect those pages and your output destination.\n"
            else:
                new_message = message + f"Page {max(0, start_page)} to {end_page} from file {file_id} processed and added to new sheet in {doc_name}.\n" if chunk_success else f"FAILED to successfully process page {max(0, start_page)} to {end_page} from file {file_id}.  Please inspect those pages and your output destination.\n"
        else: #download output
            new_message = message + f"Page {max(0, start_page)} to {end_page} from file {file_id} processed.\n" if chunk_success else f"FAILED to successfully process page {max(0, start_page)} to {end_page} from file {file_id}.  Please inspect those pages and your output destination.\n"
        logger.info(f"Message: {message}")
        return QueryResponse(
            status=current_status,
            message=new_message,
            num_images_processed=job['total_images_processed'],
        )