from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response, Depends, Request
from typing import List, Optional, Any, Union, Annotated
from pydantic import BaseModel
from app.utils.process_query_standard import process_query
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import QueryResponse, FileInfo, TruncatedSandboxResult, BatchProcessingFileInfo, SandboxResult
import json
from app.utils.file_management import temp_file_manager
import os
import logging
from app.schemas import QueryRequest
from fastapi.responses import FileResponse
import pandas as pd
from app.utils.postprocessing import handle_destination_upload, handle_download, handle_batch_destination_upload, handle_batch_chunk_result
from app.utils.data_processing import get_data_snapshot
from app.utils.preprocessing import preprocess_files
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.llm_service import get_llm_service, LLMService
from app.utils.check_connection import check_client_connection
from datetime import datetime
import time

from PyPDF2 import PdfReader
import io
# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


def construct_status_response(job: dict) -> QueryResponse:
    """Helper function to construct status response"""
    if job["status"] == "completed":
        if job["output_preferences"]["type"] == "online":
            return QueryResponse(
                result=TruncatedSandboxResult(
                    original_query=job["query"],
                    print_output="",
                    error=None,
                    timed_out=False,
                    return_value_snapshot=job["result_snapshot"]
                ),
                status="success",
                message="Processing completed",
                files=None,
                num_images_processed=job["processed_pages"]
            )
        else:  # download type
            return QueryResponse(
                result=TruncatedSandboxResult(
                    original_query=job["query"],
                    print_output="",
                    error=None,
                    timed_out=False,
                    return_value_snapshot=None
                ),
                status="success",
                message="File ready for download",
                files=[FileInfo(
                    file_path=job["result_file_path"],
                    media_type=job["result_media_type"],
                    filename=os.path.basename(job["result_file_path"]),
                    download_url=f"/download?file_path={job['result_file_path']}"
                )],
                num_images_processed=job["processed_pages"]
            )
    
    elif job["status"] == "error":
        return QueryResponse(
            result=None,
            status="error",
            message=job["error_message"],
            files=None,
            num_images_processed=job["processed_pages"]
        )
    
    else:  # processing or created
        return QueryResponse(
            result=None,
            status="processing",
            message=f"Processing in progress. {job['processed_pages']}/{job['total_pages']} pages processed",
            files=None,
            num_images_processed=job["processed_pages"]
        )


async def determine_pdf_page_count(file: UploadFile) -> int:
    """
    Determines the page count of a PDF file.
    Returns 0 if the file is not a PDF.
    """
    if not file.content_type == "application/pdf":
        return 0
        
    try:
        # Read the file into memory
        contents = await file.read()
        # Create a new BytesIO object to ensure proper handling
        pdf_stream = io.BytesIO(contents)
        pdf = PdfReader(pdf_stream)
        page_count = len(pdf.pages)
        
        # Reset file pointer for future reads
        await file.seek(0)
        
        # Close the BytesIO stream
        pdf_stream.close()
        
        return page_count
    except Exception as e:
        logger.error(f"Error determining PDF page count: {str(e)}")
        # Make sure to reset the file pointer even if there's an error
        await file.seek(0)
        return 0


@router.post("/process_query", response_model=QueryResponse)
async def process_query_entry_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    llm_service: LLMService = Depends(get_llm_service),
    json_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> QueryResponse:
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        request_data = QueryRequest(**json.loads(json_data))
        
        # Update page counts for PDF files
        if request_data.files_metadata:
            for file_meta, upload_file in zip(request_data.files_metadata, files):
                if not file_meta.page_count:  # Only update if page_count is not set
                    page_count = await determine_pdf_page_count(upload_file)
                    file_meta.page_count = page_count
                    # Reset file pointer after reading
                    await upload_file.seek(0)
        
        # Check if any PDF has more than 5 pages
        has_large_pdf = False
        total_pages = 0
        for file_meta in (request_data.files_metadata or []):
            logger.info(f"file_meta: {file_meta.page_count}")
            if file_meta.page_count and file_meta.page_count > 5:
                total_pages += file_meta.page_count
                has_large_pdf = True
                break
                
        # Decision routing based on page count
        if has_large_pdf:
            # Generate unique job ID
            logger.info(f"Large PDF detected")
            job_id = f"job_{user_id}_{int(time.time())}"
            
            # Calculate page ranges for each file
            page_chunks = []
            CHUNK_SIZE = 5  # Process 5 pages at a time
            
            for file_meta in (request_data.files_metadata or []):
                if file_meta.page_count:
                    start_page = 0
                    while start_page < file_meta.page_count:
                        end_page = min(start_page + CHUNK_SIZE, file_meta.page_count)
                        page_chunks.append(
                            BatchProcessingFileInfo(
                                file_id=file_meta.file_id or str(file_meta.name),
                                page_range=(start_page, end_page),
                                metadata=file_meta
                            ).dict()
                        )
                        start_page += CHUNK_SIZE
            
            # Initialize job in database with chunks information
            response = supabase.table("batch_jobs").insert({
                "job_id": job_id,
                "user_id": user_id,
                "status": "created",
                "total_pages": total_pages,
                "processed_pages": 0,
                "output_preferences": request_data.output_preferences.dict(),
                "created_at": datetime.utcnow().isoformat(),
                "started_at": None,
                "completed_at": None,
                "error_message": None,
                "result_snapshot": None,
                "result_file_path": None,
                "result_media_type": None,
                "page_chunks": page_chunks,
                "current_chunk": 0,
                "query": request_data.query
            }).execute()
            
            # Start first batch processing in background
            background_tasks.add_task(
                process_query_batch_endpoint,
                request=request,
                user_id=user_id,
                supabase=supabase,
                llm_service=llm_service,
                json_data=json_data,
                files=files,
                job_id=job_id
            )
            
            return QueryResponse(
                result=TruncatedSandboxResult(
                    original_query=request_data.query,
                    print_output="",
                    error=None,
                    timed_out=False,
                    return_value_snapshot=None
                ),
                status="processing",
                message=f"Batch processing initiated. Job ID: {job_id}",
                files=None,
                num_images_processed=0,
                job_id=job_id
            )
        else:
            # Route to standard processing
            return await process_query_standard_endpoint(
                request=request,
                user_id=user_id,
                supabase=supabase,
                llm_service=llm_service,
                json_data=json_data,
                files=files,
                background_tasks=background_tasks
            )
            
    except Exception as e:
        logger.error(f"Error in process_query_entry_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_query_standard", response_model=QueryResponse)
async def process_query_standard_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    llm_service: LLMService = Depends(get_llm_service),
    json_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> QueryResponse:



    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    truncated_result = None
    num_images_processed = 0

    logger.info( f"process_query_standard_endpoint called")


    try:
        request_data = QueryRequest(**json.loads(json_data))
        logger.info( f"sheet: {request_data.output_preferences.destination_url}")
        logger.info( f"sheet: {request_data.output_preferences.sheet_name}")
        logger.info(f"Processing query for user {user_id} with {len(request_data.files_metadata or [])} files")
        # Initial connection check
        await check_client_connection(request)

        load_dotenv(override=True)

        session_dir = temp_file_manager.get_temp_dir()
        logger.info("Calling preprocess_files")
        num_images_processed = 0

        try:
            preprocessed_data, num_images_processed = await preprocess_files(
                request=request,
                files=files,
                files_metadata=request_data.files_metadata,
                input_urls=request_data.input_urls,
                query=request_data.query,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                llm_service=llm_service,
                num_images_processed=num_images_processed
            )


            logger.info(f"num_images_processed: {num_images_processed}")
        except Exception as e:
            raise ValueError(e)

        # Process the query with the processed data
        sandbox = EnhancedPythonInterpreter()
        result = await process_query(
            request=request,
            query=request_data.query,
            sandbox=sandbox,
            data=preprocessed_data,
            llm_service=llm_service
        )
        await check_client_connection(request)

        result.return_value_snapshot = get_data_snapshot(result.return_value, type(result.return_value).__name__)
        logger.info(f"Query processed for user {user_id} with return value snapshot type: {type(result.return_value).__name__}")

        if result.error:
            raise HTTPException(status_code=400, detail=result.error + " -- please try rephrasing your request")
        

        truncated_result = TruncatedSandboxResult(
                original_query=result.original_query,
                print_output=result.print_output,
                error=result.error,
                timed_out=result.timed_out,
                return_value_snapshot=result.return_value_snapshot
            )


        # Handle output based on type
        if request_data.output_preferences.type == "download":
            tmp_path, media_type = handle_download(result, request_data, preprocessed_data, llm_service)
            # Add cleanup task but DON'T execute immediately
            background_tasks.add_task(temp_file_manager.cleanup_marked)
            
            # Update download URL to match client expectations
            download_url = f"/download?file_path={tmp_path}"
            

            logger.info(f"num_images_processed: {num_images_processed}")
            await check_client_connection(request)
            return QueryResponse(
                result=truncated_result,
                status="success",
                message="File ready for download",
                files=[FileInfo(
                    file_path=str(tmp_path),
                    media_type=media_type,
                    filename=os.path.basename(tmp_path),
                    download_url=download_url
                )],
                num_images_processed=num_images_processed
            )

        elif request_data.output_preferences.type == "online":
            if not request_data.output_preferences.destination_url:
                raise ValueError("destination_url is required for online type")
                
            # Handle destination sheet upload
            await handle_destination_upload(
                result.return_value,
                request_data,
                preprocessed_data,
                supabase,
                user_id,
                llm_service
            )

            # Only cleanup immediately for online type
            temp_file_manager.cleanup_marked()
            await check_client_connection(request)
            return QueryResponse(
                result=truncated_result,
                status="success",
                message="Data successfully uploaded to destination",
                files=None,
                num_images_processed=num_images_processed
            )
        
        else:
            raise ValueError(f"Invalid output type: {request_data.output_preferences.type}")

    except Exception as e:
        error_msg = str(e)[:200] if str(e).isascii() else f"Error processing request: {e.__class__.__name__}"
        logger.error(f"Process query error for user {user_id}: {error_msg}")

        return QueryResponse(
            result=truncated_result,
            status="error",
            message="An error occurred -- please try rephrasing or adding more information to your request.",
            files=None,
            num_images_processed=num_images_processed
        )
    
@router.post("/process_query/batch", response_model=QueryResponse)
async def process_query_batch_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    job_id: str,
    llm_service: LLMService = Depends(get_llm_service),
    json_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> QueryResponse:
    """Handles batch processing of large document sets asynchronously."""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        request_data = QueryRequest(**json.loads(json_data))
        session_dir = temp_file_manager.get_temp_dir()
        
        # Remove await for Supabase operations
        job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).single().execute()
        if not job_response.data:
            raise ValueError("Job not found")
            
        job_data = job_response.data
        current_chunk = job_data["current_chunk"]
        page_chunks = job_data["page_chunks"]
        
        logger.info(f"Batch processing chunk: {current_chunk}")
        
        if current_chunk >= len(page_chunks):
            raise ValueError("No more chunks to process")
            
        current_processing_info = BatchProcessingFileInfo(**page_chunks[current_chunk])
        
        try:
            # Preprocess and process chunk
            preprocessed_data, num_images_processed = await preprocess_files(
                request=request,
                files=files,
                files_metadata=[current_processing_info.metadata],
                input_urls=request_data.input_urls,
                query=request_data.query,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                llm_service=llm_service,
                num_images_processed=0,
                page_range=current_processing_info.page_range
            )

            sandbox = EnhancedPythonInterpreter()
            result = await process_query(
                request=request,
                query=request_data.query,
                sandbox=sandbox,
                data=preprocessed_data,
                llm_service=llm_service
            )

            if result.error:
                raise ValueError(result.error)

            # Handle post-processing
            status, result_file_path, result_media_type = await handle_batch_chunk_result(
                result=result,
                request_data=request_data,
                preprocessed_data=preprocessed_data,
                supabase=supabase,
                user_id=user_id,
                llm_service=llm_service,
                job_id=job_id,
                session_dir=session_dir,
                current_chunk=current_chunk,
                total_chunks=len(page_chunks),
                num_images_processed=num_images_processed
            )

            # Cleanup temporary files for this chunk
            temp_file_manager.cleanup_marked()

            # Trigger next chunk if not done
            if current_chunk + 1 < len(page_chunks):
                logger.info(f"Triggering next chunk: {current_chunk + 1}/{len(page_chunks)}")
                background_tasks.add_task(
                    process_query_batch_endpoint,
                    request=request,
                    user_id=user_id,
                    supabase=supabase,
                    llm_service=llm_service,
                    json_data=json_data,
                    files=files,
                    job_id=job_id
                )

            return QueryResponse(
                result=TruncatedSandboxResult(
                    original_query=request_data.query,
                    print_output="",
                    error=None,
                    timed_out=False,
                    return_value_snapshot=result.return_value_snapshot
                ),
                status=status,
                message=f"Batch {current_chunk + 1}/{len(page_chunks)} processed successfully",
                files=None,
                num_images_processed=num_images_processed,
                job_id=job_id
            )

        except Exception as e:
            # Remove await for Supabase error update
            supabase.table("batch_jobs").update({
                "status": "error",
                "error_message": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }).eq("job_id", job_id).execute()
            raise

    except Exception as e:
        logger.error(f"Batch processing error for job {job_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_query/status", response_model=QueryResponse)
async def process_query_status_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    job_id: str = Form(...),
) -> QueryResponse:
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    try:
        # Remove cache-related code
        job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = job_response.data[0]
        
        # Construct response based on job status
        return construct_status_response(job)
            
    except Exception as e:
        logger.error(f"Error in process_query_status_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

