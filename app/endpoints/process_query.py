from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Response
from typing import List, Annotated, Optional
from app.utils.process_query_algo import process_query_algo
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import QueryResponse, FileInfo, TruncatedSandboxResult, BatchProcessingFileInfo, ChunkResponse, FileDataInfo
import json
from app.utils.s3_file_management import temp_file_manager
from app.utils.s3_file_actions import s3_file_actions
import os
import logging
from app.schemas import QueryRequest
from app.utils.postprocessing import handle_destination_upload, handle_download, handle_batch_chunk_result
from app.utils.data_processing import get_data_snapshot
from app.utils.preprocessing import preprocess_files, determine_pdf_page_count, pdf_classifier, check_limits_pre_batch
from app.utils.controllers.process_batch_chunk import process_batch_chunk
from app.utils.controllers.standard_process_controller import process_query_standard
from app.utils.controllers.batch_process_controller import process_query_batch
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.message_builder import check_client_connection, construct_status_response_batch, construct_status_response_standard
from app.schemas import InputUrl
from datetime import datetime, UTC
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
import asyncio
from app.utils.s3_file_actions import S3PDFStreamer
from app.dev_utils.memory_profiler import profile_memory


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/process_query", response_model=QueryResponse)
@profile_memory
async def process_query_entry_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    json_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> QueryResponse:
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    try:
        # Add debug logging
        form = await request.form()
        logger.info(f"Form data keys: {form.keys()}")
        logger.info(f"Form files: {[k for k in form.keys() if k.startswith('files')]}")
        
        request_data = QueryRequest(**json.loads(json_data))
        logger.info(f"Number of in memory files received: {len(files)}")
        logger.info(f"Number of file metadata entries: {len(request_data.files_metadata)}")
        
        # Store file contents in memory before background processing
        files_data = []
        i = 0
        for file_meta in request_data.files_metadata:
            if file_meta.s3_key:  # Explicit check for S3 files
                # For S3 files, just store the metadata - no content loading
                files_data.append({
                    'content': file_meta.s3_key,  # No content for S3 files
                    'filename': file_meta.name,
                    'content_type': file_meta.type
                })
            else:
                # Handle regular files using the index from metadata
    
                if not hasattr(file_meta, 'index'):
                    raise HTTPException(status_code=400, detail=f"Missing index for uploaded file {file_meta.name}")
                try:
                    file = files[i]
                    content = await file.read()
                    await file.seek(0)  # Reset file pointer after reading
                    files_data.append({
                        'content': content,
                        'filename': file.filename,
                        'content_type': file.content_type
                    })
                    i += 1
                except IndexError:
                    raise HTTPException(status_code=400, detail=f"No uploaded file found at index {file_meta.index} for {file_meta.name}")
        
                    # Create new UploadFile objects for the background task
        
        upload_files = []
        for file_data in files_data:
            file_obj = None if isinstance (file_data['content'], str) else io.BytesIO(file_data['content'])
            upload_file = UploadFile(
                file=file_obj,
                filename=file_data['filename'],
                headers={'content-type': file_data['content_type']}
            )
            upload_files.append(upload_file)

        logger.info(f"LENGTH of upload_files: {len(upload_files)}")
        total_pages = 0
        # Update page counts
        for i, file_meta in enumerate(request_data.files_metadata):
            if not file_meta.page_count and file_meta.type == 'application/pdf':
                if file_meta.s3_key:
                    try:
                        # Use S3PDFStreamer for more reliable page counting
                        pdf_streamer = S3PDFStreamer(s3_file_actions.s3_client, s3_file_actions.bucket, file_meta.s3_key)
                        file_meta.page_count = pdf_streamer.page_count
                        total_pages += file_meta.page_count
                    except Exception as e:
                        logger.error(f"Error getting page count for S3 PDF {file_meta.s3_key}: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to process PDF file {file_meta.name}: {str(e)}"
                        )
                else:
                    try:
                        file_content = io.BytesIO(files_data[i]['content'])
                        pdf_reader = PdfReader(file_content)
                        file_meta.page_count = len(pdf_reader.pages)
                        total_pages += file_meta.page_count
                        file_content.seek(0)  # Reset file pointer after reading
                    except Exception as e:
                        logger.error(f"Error getting page count for uploaded PDF {file_meta.name}: {str(e)}")
                        raise HTTPException(
                            status_code=500,
                            detail=f"Failed to process PDF file {file_meta.name}: {str(e)}"
                        )

        #If destination url and no input url, add destination url and sheet name to input urls for processing context
        if request_data.output_preferences.destination_url and not request_data.input_urls and request_data.output_preferences.modify_existing:
            request_data.input_urls = [InputUrl(url=request_data.output_preferences.destination_url, sheet_name=request_data.output_preferences.sheet_name)]
          
        
        # Check if any PDF has more than CHUNK_SIZE pages
        need_to_batch = False
        contains_image_or_like = False 
        
        # Check for images to propagate flag to llm service for prompt selection
        for i, file_data in enumerate(files_data):
            file_meta = request_data.files_metadata[i]  # Get corresponding metadata
            if file_data['content_type'].startswith('image/'):
                contains_image_or_like = True
            elif file_data['content_type'] == 'application/pdf':
                if file_meta.s3_key:
                    content = file_meta  # Pass the metadata object directly
                else:
                    # Create a BytesIO object for the PDF content
                    content = io.BytesIO(file_data['content'])
                is_image_like = await pdf_classifier(content)
                if not file_meta.s3_key:
                    content.seek(0)  # Reset file pointer after classification
                contains_image_or_like = contains_image_or_like or is_image_like
                if is_image_like and file_meta.page_count > int(os.getenv("CHUNK_SIZE")):
                    need_to_batch = True
                    break
        
        logger.info(f"contains_image_or_like: {contains_image_or_like}")

        # Generate unique job ID
        job_id = request_data.job_id
        
        # Initialize job in database (both batch and standard)
        
        if not need_to_batch:
            supabase.table("jobs").update({
                "type": "standard",
                "status": "processing"
            }).eq("job_id", job_id).execute() 


        # Decision routing based on page count and content type
        if need_to_batch:
            
            # Calculate page ranges for each file
            page_chunks = []
            CHUNK_SIZE = int(os.getenv("CHUNK_SIZE")) 
            logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
            
            for file_meta in request_data.files_metadata:
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
            
            chunk_status_default_list = []
            for i in range(len(page_chunks)):
                chunk_status_default_list.append(f"Chunk {i+1} Status:")

            supabase.table("jobs").update({
                "status": "processing",
                "total_pages": total_pages,
                "processed_pages": 0,
                "started_at": datetime.now(UTC).isoformat(),
                "page_chunks": page_chunks,
                "current_chunk": 0,
                "query": request_data.query,
                "chunk_status": chunk_status_default_list,
                "type": "batch"
            }).eq("job_id", job_id).execute()   
            logger.info("Large PDF detected")
            try:
                await check_limits_pre_batch(supabase, user_id, total_pages)
            except ValueError as e:
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message=str(e),
                    files=None,
                    num_images_processed=0,
                    error=str(e),
                    total_pages=0
                )
                
            logger.info(f"LENGTH of upload_files: {len(upload_files)}")
            # Start first batch processing in background
            batch_response = await process_query_batch(
                    request=request,
                    user_id=user_id,
                    supabase=supabase,
                    request_data=request_data,
                    files=upload_files,
                    job_id=job_id,
                    contains_image_or_like=contains_image_or_like
            )
            
            return batch_response
        else:
            # Route to standard processing
            return await process_query_standard(
                request=request,
                user_id=user_id,
                supabase=supabase,
                request_data=request_data,
                files=upload_files,
                background_tasks=background_tasks,
                contains_image_or_like=contains_image_or_like,
                job_id=job_id
            )
            
    except Exception as e:
        logger.error(f"Error in process_query_entry_endpoint: {str(e)}")
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
        
    await check_client_connection(request)
    try:
        # Remove cache-related code
        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = job_response.data[0]
        if job["type"] == "batch":
            response = construct_status_response_batch(job)
        else:
            response = construct_status_response_standard(job)
        supabase.table("jobs").update({
            "message": response.message
        }).eq("job_id", job_id).execute()
        logger.info(f"response: {response}")

        return response
            
    except Exception as e:
        logger.error(f"Error in process_query_status_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



