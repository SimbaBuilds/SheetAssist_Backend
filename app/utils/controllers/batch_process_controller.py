from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Response
from typing import List, Annotated, Optional
from app.schemas import QueryResponse
import json
from app.utils.s3_file_management import temp_file_manager
from app.utils.s3_file_actions import s3_file_actions
import os
import logging
from app.schemas import QueryRequest
from app.utils.controllers.process_batch_chunk import process_batch_chunk
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.message_builder import check_client_connection, construct_status_response_batch
from app.schemas import InputUrl
from datetime import datetime, UTC
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
import asyncio
from app.utils.s3_file_actions import S3PDFStreamer


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_query_batch(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    request_data: QueryRequest,
    job_id: str,
    files: List[UploadFile] = File(default=[]),
    contains_image_or_like: bool = False
) -> QueryResponse:
    """Handles batch processing of large document sets synchronously."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Initialize job with processing status
        supabase.table("jobs").update({
            "started_at": datetime.now(UTC).isoformat(),
            "total_images_processed": 0,
            "processed_pages": 0,
        }).eq("job_id", job_id).execute()

        session_dir = temp_file_manager.get_temp_dir()
        total_images_processed = 0
        previous_chunk_return_value = None

        # Get job data
        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
        if not job_response.data or len(job_response.data) == 0:
            raise ValueError("Job not found")
            
        job_data = job_response.data[0]
        page_chunks = job_data["page_chunks"]
        response = None

        message = ''
        message = construct_status_response_batch(job_data)
        supabase.table("jobs").update({"message": message}).eq("job_id", job_id).execute()
        

        # Process all chunks sequentially
        for chunk_index in range(len(page_chunks)):
            try:
                await check_client_connection(request)
                logger.info(f"Processing chunk {chunk_index + 1} of {len(page_chunks)}")
                
                # Update job with current chunk status
                supabase.table("jobs").update({
                    "current_chunk": chunk_index,
                }).eq("job_id", job_id).execute()

                # Process the chunk for ChunkResponse
                response = await process_batch_chunk(
                    user_id=user_id,
                    supabase=supabase,
                    request_data=request_data,
                    files=files,
                    job_id=job_id,
                    session_dir=session_dir,
                    current_chunk=chunk_index,
                    previous_chunk_return_value=previous_chunk_return_value,
                    contains_image_or_like=contains_image_or_like
                )

                # Store the return value for the next chunk
                logger.info(f"Storing previous chunk return value with type: {type(response.result.return_value)} and first element: {response.result.return_value[0]}")
                previous_chunk_return_value = response.result.return_value #tuple

                # Get updated job data
                job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
                if not job_response.data or len(job_response.data) == 0:
                    raise ValueError("Job not found")
                job_data = job_response.data[0]
                
                # Accumulate total images processed
                chunk_status = job_data.get("chunk_status", [])
                current_chunk_status = chunk_status[chunk_index]
                if "Success" in current_chunk_status:
                    total_images_processed += response.num_images_processed

                logger.info(f"TOTAL images processed: {total_images_processed}")
                
                # Calculate progress
                processed_pages = min((chunk_index + 1) * int(os.getenv("CHUNK_SIZE")), job_data["total_pages"])
                is_last_chunk = chunk_index == len(page_chunks) - 1
                
                message = construct_status_response_batch(job_data)
                # Comprehensive job update after each chunk
                update_data = {
                    "current_chunk": min(chunk_index + 1, len(page_chunks) - 1),
                    "processed_pages": processed_pages,
                    "total_images_processed": total_images_processed,
                    "result_snapshot": response.result.return_value_snapshot if is_last_chunk else None,
                    "message": message
                }
                
                if is_last_chunk:
                    update_data.update({
                        "completed_at": datetime.now(UTC).isoformat(),
                    })
                
                supabase.table("jobs").update(update_data).eq("job_id", job_id).execute()

            except ValueError as e:
                # Update job with error status but don't raise exception yet
                error_msg = str(e)
                supabase.table("jobs").update({
                    "status": "error",
                    "error_message": error_msg,
                    "message": error_msg,
                    "completed_at": datetime.now(UTC).isoformat(),
                }).eq("job_id", job_id).execute()
                
                # Return error response instead of raising exception
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message=error_msg,
                    files=None,
                    num_images_processed=total_images_processed,
                    job_id=job_id,
                    error=error_msg
                )

        # Final status update based on chunk statuses
        chunk_status = job_data.get("chunk_status", [])
        all_chunk_statuses_str = ", ".join(chunk_status)
        final_status = "completed_with_error(s)" if "Error" in all_chunk_statuses_str else "completed"
        
        # Get updated job data for message generation
        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
        job = job_response.data[0]
        job["status"] = final_status # local job update to get completion message before supabase status update and end
        message = construct_status_response_batch(job)

        supabase.table("jobs").update({
            "message": message
        }).eq("job_id", job_id).execute()       
        
        
        supabase.table("jobs").update({
            "status": final_status,
            "completed_at": datetime.now(UTC).isoformat(),
        }).eq("job_id", job_id).execute()

    
        return QueryResponse(
            original_query=request_data.query,
            status=response.status,
            message=message,
            files=response.files,
            num_images_processed=total_images_processed,
            job_id=response.job_id
        )

    except Exception as e:
        error_msg = str(e)[:200] if str(e).isascii() else f"Error processing request: {e.__class__.__name__}"
        logger.error(f"Batch processing error for job {job_id}: {error_msg}")
        
        # Comprehensive error update
        supabase.table("jobs").update({
            "status": "error",
            "error_message": error_msg,
            "message": error_msg,
            "completed_at": datetime.now(UTC).isoformat(),
            "total_images_processed": total_images_processed if 'total_images_processed' in locals() else 0,
        }).eq("job_id", job_id).execute()
        
        # Return error response instead of raising exception
        return QueryResponse(
            original_query=request_data.query,
            status="error",
            message=error_msg,
            files=None,
            num_images_processed=total_images_processed if 'total_images_processed' in locals() else 0,
            job_id=job_id,
            error=error_msg
        )

