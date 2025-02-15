from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Response
from typing import List, Annotated, Optional
from app.utils.process_query_algo import process_query_algo
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import QueryResponse, FileInfo
import json
from app.utils.s3_file_management import temp_file_manager
import os
import logging
from app.schemas import QueryRequest
from app.utils.postprocessing import handle_destination_upload, handle_download
from app.utils.data_processing import get_data_snapshot
from app.utils.preprocessing import preprocess_files
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.message_builder import check_client_connection, construct_status_response_standard
from dotenv import load_dotenv
import io
from datetime import datetime, UTC


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_query_standard(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    request_data: QueryRequest,
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    contains_image_or_like: bool = False,
    job_id: str = None
) -> QueryResponse:
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    await check_client_connection(request)
    
    try:
        # Update job to processing status
        supabase.table("jobs").update({
            "type": "standard",
            "status": "processing",
            "started_at": datetime.utcnow().isoformat()
        }).eq("job_id", job_id).execute()

        # Get job data for message
        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
        job = job_response.data[0]
        message = construct_status_response_standard(job)
        
        supabase.table("jobs").update({
            "message": message
        }).eq("job_id", job_id).execute()

        # Create session directory
        session_dir = temp_file_manager.get_temp_dir()
        logger.info("Calling preprocess_files")
        num_images_processed = 0

        # Store file contents in memory if not S3
        files_data = []
        for file_meta in request_data.files_metadata:
            try:
                if file_meta.s3_key:
                    # For S3 files, just store the metadata - no content loading
                    files_data.append({
                        'content': None,  # No content for S3 files
                        'filename': file_meta.name,
                        'content_type': file_meta.type
                    })
                else:
                    # Handle regular files
                    file = files[file_meta.index]
                    content = await file.read()
                    await file.seek(0)
                    files_data.append({
                        'content': content,
                        'filename': file.filename,
                        'content_type': file.content_type
                    })
            except Exception as e:
                logger.error(f"Error reading file {file_meta.name}: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")

        # Create new UploadFile objects for preprocessing
        upload_files = []
        for file_data in files_data:
            if file_data['content'] is None:
                # Skip S3 files - they'll be handled by their metadata
                continue
            file_obj = io.BytesIO(file_data['content'])
            upload_file = UploadFile(
                file=file_obj,
                filename=file_data['filename'],
                headers={'content-type': file_data['content_type']}
            )
            upload_files.append(upload_file)

        supabase.table("jobs").update({
        "status": "processing"
            }).eq("job_id", job_id).execute()

        # logger.info(f"upload_files: {upload_files}\n request_data.files_metadata: {request_data.files_metadata}")
        try:
            preprocessed_data, num_images_processed = await preprocess_files(
                files=upload_files,
                files_metadata=request_data.files_metadata,
                input_urls=request_data.input_urls,
                query=request_data.query,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                num_images_processed=num_images_processed
            )
            logger.info(f" -- Returning preprocessed_data: {preprocessed_data} and num_images_processed: {num_images_processed} --- ")
        except ValueError as e:
            # Specifically handle image/overage limit errors
            error_msg = str(e)
            if "limit reached" in error_msg.lower():
                message="Usage or overage limit reached. Please check account settings.",
                supabase.table("jobs").update({
                    "message": message
                }).eq("job_id", job_id).execute()
                message="Usage or overage limit reached. Please check account settings.",
                supabase.table("jobs").update({
                    "status": "error"
                }).eq("job_id", job_id).execute()
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message="Limit reached. Please check your account settings.",
                    files=None,
                    num_images_processed=0,
                    error=error_msg,
                    total_pages=0
                )
            raise  # Re-raise other ValueError exceptions

        # Process the query with the processed data
        sandbox = EnhancedPythonInterpreter()
        result = await process_query_algo(
            request=request,
            query=request_data.query,
            sandbox=sandbox,
            data=preprocessed_data,
            contains_image_or_like=contains_image_or_like
        )
        await check_client_connection(request)
        logger.info(f"result: {result}")
        result.return_value_snapshot = get_data_snapshot(result.return_value, type(result.return_value).__name__)
        logger.info(f"Query processed for user {user_id} with return value snapshot type: {type(result.return_value).__name__}")

        if result.error:
            message = "There was an error processing your request. This application may not have the ability to complete your request. You can also try rephrasing your request or breaking it down into multiple requests."
            # Update job with error status
            supabase.table("jobs").update({
                "message": message,
                }).eq("job_id", job_id).execute()
            supabase.table("jobs").update({
                "status": "error",
                "completed_at": datetime.utcnow().isoformat(),
                "error_message": result.error,
                "total_images_processed": 0
            }).eq("job_id", job_id).execute()
            
            return QueryResponse(
                original_query=request_data.query,
                status="error",
                message=message,
                files=None,
                num_images_processed=0,
                error=result.error,
                total_pages=0
            )
        
        await check_client_connection(request)
        
        # Handle output based on type
        if request_data.output_preferences.type == "download":
            try:
                tmp_path, media_type = await handle_download(result, request_data, preprocessed_data)
                
                # # Add cleanup task but DON'T execute immediately
                # asyncio.create_task(temp_file_manager.cleanup_marked)
                
                # Update download URL to match client expectations
                download_url = f"/download?file_path={tmp_path}"
                
                # Update job with success status for download
                supabase.table("jobs").update({
                    "result_snapshot": result.return_value_snapshot,
                    "result_file_path": str(tmp_path),
                    "total_images_processed": num_images_processed
                }).eq("job_id", job_id).execute()

                # Get updated job data for message
                job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
                job = job_response.data[0]
                job["status"] = "completed" # local job update to get completion message before supabase status update and end
                message = construct_status_response_standard(job)

                supabase.table("jobs").update({
                    "message": message
                }).eq("job_id", job_id).execute()       
                
                
                supabase.table("jobs").update({
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                }).eq("job_id", job_id).execute()                
                        
                return QueryResponse(
                    original_query=request_data.query,
                    status="completed",
                    message=message,
                    files=[FileInfo(
                        file_path=str(tmp_path),
                        media_type=media_type,
                        filename=os.path.basename(tmp_path),
                        download_url=download_url
                    )],
                    num_images_processed=num_images_processed
                )
            except Exception as e:
                logger.error(f"Download processing failed for user {user_id}: {str(e)}")
                error_message = f"Failed to process download: {str(e)[:100]}..."
                # Update job with error status
                supabase.table("jobs").update({
                    "status": "error",
                    "completed_at": datetime.utcnow().isoformat(),
                    "error_message": str(e),
                    "total_images_processed": 0
                }).eq("job_id", job_id).execute()
                
                # Get updated job data for message
                job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
                job = job_response.data[0]
                message = construct_status_response_standard(job)
                supabase.table("jobs").update({
                    "message": message
                }).eq("job_id", job_id).execute()           
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message=message,
                    files=None,
                    num_images_processed=0,
                    error=str(e),
                    total_pages=0
                )

        elif request_data.output_preferences.type == "online":
            if not request_data.output_preferences.destination_url:
                raise ValueError("destination_url is required for online type")
                
            try:
                # Handle destination sheet upload
                await handle_destination_upload(
                    result.return_value,
                    request_data,
                    preprocessed_data,
                    supabase,
                    user_id
                )
                supabase.table("jobs").update({
                    "completed_at": datetime.utcnow().isoformat(),
                    "result_snapshot": result.return_value_snapshot,
                    "total_images_processed": num_images_processed
                }).eq("job_id", job_id).execute()

                # Get updated job data for message
                job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
                job = job_response.data[0]
                job["status"] = "completed" # local job update to get completion message before supabase status update and end
                message = construct_status_response_standard(job)

                supabase.table("jobs").update({
                    "message": message
                }).eq("job_id", job_id).execute()       
                
                
                supabase.table("jobs").update({
                    "status": "completed",
                    "completed_at": datetime.now(UTC).isoformat(),
                }).eq("job_id", job_id).execute()  
                
                
                
                
                # Only cleanup immediately for online type
                await temp_file_manager.cleanup_marked()
                await check_client_connection(request)

                return QueryResponse(
                    original_query=request_data.query,
                    status="completed",
                    message=message,
                    files=None,
                    num_images_processed=num_images_processed
                )
            except Exception as e:
                logger.error(f"Online destination upload failed for user {user_id}: {str(e)}")
                error_message = "Failed to upload to destination."
                # Update job with error status
                supabase.table("jobs").update({
                    "status": "error",
                    "completed_at": datetime.utcnow().isoformat(),
                    "error_message": str(e),
                    "total_images_processed": num_images_processed
                }).eq("job_id", job_id).execute()
                
                # Get updated job data for message
                job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
                job = job_response.data[0]
                message = construct_status_response_standard(job)
                supabase.table("jobs").update({
                    "message": message
                }).eq("job_id", job_id).execute()                
                return QueryResponse(
                    original_query=request_data.query,
                    status="error", 
                    message=message,
                    files=None,
                    num_images_processed=0,
                    error=str(e),
                    total_pages=0
                )
        
        else:
            raise ValueError(f"Invalid output type: {request_data.output_preferences.type}")

    except Exception as e:
        error_msg = str(e)[:200] if str(e).isascii() else f"Error processing request: {e.__class__.__name__}"
        logger.error(f"Process query error for user {user_id}: {error_msg}")
        
        # Update job with error status for any uncaught exceptions
        supabase.table("jobs").update({
            "status": "error",
            "completed_at": datetime.utcnow().isoformat(),
            "error_message": error_msg,
            "total_images_processed": 0
        }).eq("job_id", job_id).execute()
        
        # Get updated job data for message
        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
        job = job_response.data[0]
        message = construct_status_response_standard(job)
        supabase.table("jobs").update({
            "message": message
        }).eq("job_id", job_id).execute()
        return QueryResponse(
            original_query=request_data.query,
            status="error",
            message=message,
            files=None,
            num_images_processed=0,
            error=str(e),
            total_pages=0
        )
