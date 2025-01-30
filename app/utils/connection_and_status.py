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

def prevent_duplicate_message(existing_message: str, new_line: str) -> str:
    """
    Checks if the new line is already the last line in the existing message.
    Returns the combined message without duplication.
    """
    if not existing_message:
        return new_line
    
    # Split existing message into lines and get the last non-empty line
    existing_lines = [line.strip() for line in existing_message.split('\n') if line.strip()]
    if existing_lines and existing_lines[-1] == new_line.strip():
        return existing_message
    
    return existing_message + new_line

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
        existing_message = job.get("message", "")
        if output_preferences['type'] == 'online':
            new_message = existing_message + f"Processing complete."

            return QueryResponse(
                status=current_status,
                message=new_message,
                num_images_processed=job["total_images_processed"],
            )
        else:  # download type
            new_message = existing_message + f"Processing complete. Your file should download automatically."
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
            message=f"Processing pages {page_chunks[current_chunk]['page_range'][0] + 1} to {page_chunks[current_chunk]['page_range'][1]}.\n",
            num_images_processed=job["total_images_processed"],
        )
    
    elif current_status == "processing":        
        completed_chunk = int(job.get('current_chunk', 0)) - 1 # -1 because current_chunk already got incremented
        completed_chunk = max(0, completed_chunk)
        start_page = int(page_chunks[completed_chunk]['page_range'][0]) + 1
        end_page = int(page_chunks[completed_chunk]['page_range'][1])
        chunk_status = job.get("chunk_status", [])
        chunk_success = "Success" in chunk_status[completed_chunk] if completed_chunk < len(chunk_status) else False
        logger.info(f"CHUNK STATUS: {chunk_status[completed_chunk]}, Current chunk: {completed_chunk}, Chunk SUCCESS: {chunk_success}")
        
        # FIX: Use job's existing message as base for new_message
        existing_message = job.get("message", "")
        
        # Construct the new line based on output preferences and success status
        if output_preferences['type'] == 'online':
            if output_preferences['modify_existing']:
                new_line = f"Page {max(0, start_page)} to {end_page} from file {file_id} processed and appended to {doc_name} - {sheet_name}.\n" if chunk_success else f"FAILED to successfully process page {max(0, start_page)} to {end_page} from file {file_id}.  Please inspect problematic pages and your output destination and try again.\n"
            else:
                new_line = f"Page {max(0, start_page)} to {end_page} from file {file_id} processed and added to new sheet in {doc_name}.\n" if chunk_success else f"FAILED to successfully process page {max(0, start_page)} to {end_page} from file {file_id}.  Please inspect problematic pages and your output destination and try again.\n"
        else: #download output
            new_line = f"Page {max(0, start_page)} to {end_page} from file {file_id} processed.\n" if chunk_success else f"FAILED to successfully process page {max(0, start_page)} to {end_page} from file {file_id}.  Please inspect problematic pages and your output destination and try again.\n"        
        # Combine messages without duplication
        new_message = prevent_duplicate_message(existing_message, new_line)
        
        logger.info(f"Updated message: {new_message}")
        return QueryResponse(
            status=current_status,
            message=new_message,
            num_images_processed=job['total_images_processed'],
        )