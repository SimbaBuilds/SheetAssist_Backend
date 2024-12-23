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
from app.utils.postprocessing import handle_destination_upload, handle_download
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
from fastapi_cache import FastAPICache
from fastapi_cache.decorator import cache
from fastapi_cache.backends.redis import RedisBackend
from redis import asyncio as aioredis
from contextlib import asynccontextmanager
from fastapi import FastAPI
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup
    redis = aioredis.from_url(os.getenv("REDIS_URL"), encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    yield
    # Cleanup (if needed)
    await redis.close()


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
        
        # Count total pages from all files
        total_pages = 0
        for file_meta in (request_data.files_metadata or []):
            if file_meta.page_count:
                total_pages += file_meta.page_count
                
        # Decision routing based on page count
        if total_pages > 5:
            # Generate unique job ID
            job_id = f"job_{user_id}_{int(time.time())}"
            
            # Initialize job in database
            await supabase.table("batch_jobs").insert({
                "job_id": job_id,
                "user_id": user_id,
                "status": "created",
                "total_pages": total_pages,
                "processed_pages": 0,
                "output_preferences": request_data.output_preferences.dict(),
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            # Start batch processing in background
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
            await check_client_connection(request)


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
                            file_id=file_meta.file_id,
                            page_range=(start_page, end_page),
                            metadata=file_meta
                        )
                    )
                    start_page += CHUNK_SIZE
        
        # Store page chunks information in the job record
        await supabase.table("batch_jobs").update({
            "status": "processing",
            "started_at": datetime.utcnow().isoformat(),
            "page_chunks": [chunk.dict() for chunk in page_chunks],
            "current_chunk": 0
        }).eq("job_id", job_id).execute()

        # Preprocess files
        try:
            # Get current chunk information
            job_data = await supabase.table("batch_jobs").select("*").eq("job_id", job_id).single().execute()
            current_chunk = job_data.data["current_chunk"]
            
            if current_chunk >= len(page_chunks):
                raise ValueError("No more chunks to process")
                
            current_processing_info = page_chunks[current_chunk]
            
            # Use the processing info for preprocessing
            preprocessed_data, num_images_processed = await preprocess_files(
                request=request,
                files=files,
                files_metadata=[current_processing_info.metadata],  # Pass original metadata
                input_urls=request_data.input_urls,
                query=request_data.query,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                llm_service=llm_service,
                num_images_processed=0,
                page_range=current_processing_info.page_range  # Pass page range separately
            )
            
            # Update processed pages count and current chunk
            await supabase.table("batch_jobs").update({
                "processed_pages": job_data.data["processed_pages"] + num_images_processed,
                "current_chunk": current_chunk + 1
            }).eq("job_id", job_id).execute()

        except Exception as e:
            await supabase.table("batch_jobs").update({
                "status": "error",
                "error_message": str(e),
                "completed_at": datetime.utcnow().isoformat()
            }).eq("job_id", job_id).execute()
            raise

        # Process the query
        sandbox = EnhancedPythonInterpreter()
        result = await process_query(
            request=request,
            query=request_data.query,
            sandbox=sandbox,
            data=preprocessed_data,
            llm_service=llm_service
        )

        if result.error:
            await supabase.table("batch_jobs").update({
                "status": "error",
                "error_message": result.error,
                "completed_at": datetime.utcnow().isoformat()
            }).eq("job_id", job_id).execute()
            raise ValueError(result.error)

        # Get current job data
        job_data = await supabase.table("batch_jobs").select("*").eq("job_id", job_id).single().execute()
        
        # Handle output based on preferences
        if request_data.output_preferences.type == "download":
            # Store intermediate results for later combination
            chunk_results_path = os.path.join(session_dir, f"chunk_{job_data.data['current_chunk']}.pkl")
            result.return_value.to_pickle(chunk_results_path)
            
            # If this is the last chunk, combine all results
            if job_data.data["current_chunk"] == len(job_data.data["page_chunks"]) - 1:
                combined_df = pd.DataFrame()
                for i in range(len(job_data.data["page_chunks"])):
                    chunk_path = os.path.join(session_dir, f"chunk_{i}.pkl")
                    chunk_df = pd.read_pickle(chunk_path)
                    combined_df = pd.concat([combined_df, chunk_df], ignore_index=True)
                
                # Generate final downloadable file
                tmp_path, media_type = await handle_download(
                    SandboxResult(return_value=combined_df, error=None),
                    request_data,
                    preprocessed_data,
                    llm_service
                )
                
                await supabase.table("batch_jobs").update({
                    "status": "completed",
                    "result_file_path": str(tmp_path),
                    "result_media_type": media_type,
                    "completed_at": datetime.utcnow().isoformat()
                }).eq("job_id", job_id).execute()

        elif request_data.output_preferences.type == "online":
            # For online output, handle each chunk immediately
            await handle_destination_upload(
                result.return_value,
                request_data,
                preprocessed_data,
                supabase,
                user_id,
                llm_service
            )
            
            if job_data.data["current_chunk"] == len(job_data.data["page_chunks"]) - 1:
                await supabase.table("batch_jobs").update({
                    "status": "completed",
                    "completed_at": datetime.utcnow().isoformat()
                }).eq("job_id", job_id).execute()

        # Cleanup temporary files
        temp_file_manager.cleanup_marked()

        # Return success response
        return QueryResponse(
            result=TruncatedSandboxResult(
                original_query=request_data.query,
                print_output="",
                error=None,
                timed_out=False,
                return_value_snapshot=None
            ),
            status="success",
            message=f"Batch processing completed for job {job_id}",
            files=None,
            num_images_processed=num_images_processed,
            job_id=job_id
        )

    except Exception as e:
        logger.error(f"Batch processing error for job {job_id}: {str(e)}")
        # Update job status to error if not already done
        await supabase.table("batch_jobs").update({
            "status": "error",
            "error_message": str(e),
            "completed_at": datetime.utcnow().isoformat()
        }).eq("job_id", job_id).execute()
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process_query/status", response_model=QueryResponse)
@cache(expire=10)  # Cache for 10 seconds
async def process_query_status_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    job_id: str = Form(...),
) -> QueryResponse:
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    try:
        # First check cache for active jobs
        cache_key = f"job_status:{job_id}"
        cached_status = await FastAPICache.get(cache_key)
        
        if cached_status and cached_status["status"] != "completed":
            return QueryResponse(**cached_status)
            
        # If not in cache or job completed, check database
        job_response = await supabase.table("batch_jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = job_response.data[0]
        
        # Construct response based on job status
        response = construct_status_response(job)
        
        # Cache response if job is still processing
        if job["status"] in ["created", "processing"]:
            await FastAPICache.set(cache_key, response.dict(), expire=10)
            
        return response
            
    except Exception as e:
        logger.error(f"Error in process_query_status_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

