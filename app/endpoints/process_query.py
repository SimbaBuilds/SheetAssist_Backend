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
from app.utils.process_batch_chunk import process_batch_chunk
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.connection_and_status import check_client_connection, construct_status_response_batch, construct_status_response_standard
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

router = APIRouter()

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

        # Process all chunks sequentially
        for chunk_index in range(len(page_chunks)):
            await check_client_connection(request)
            logger.info(f"Processing chunk {chunk_index + 1} of {len(page_chunks)}")
            

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
            previous_chunk_return_value = response.result.return_value #tuple

                    # Get job data
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
            # Update progress including images processed
            processed_pages = min((chunk_index + 1) * int(os.getenv("CHUNK_SIZE")), job_data["total_pages"])
            supabase.table("jobs").update({
                "current_chunk": min(chunk_index + 1, len(page_chunks) - 1),
                "processed_pages": processed_pages,
                "total_images_processed": total_images_processed,
                "status": "processing" if chunk_index < len(page_chunks) - 1 else "completed",
                "completed_at": datetime.now(UTC).isoformat() if chunk_index == len(page_chunks) - 1 else None
            }).eq("job_id", job_id).execute()

        chunk_status = job_data.get("chunk_status", [])
        all_chunk_statuses_str = ", ".join(chunk_status)
        if "Error" in all_chunk_statuses_str:
            supabase.table("jobs").update({
                "status": "completed_with_error(s)",
            }).eq("job_id", job_id).execute()
                
        return QueryResponse(
            original_query=request_data.query,
            status=response.status,
            message=response.message,
            files=response.files,
            num_images_processed=total_images_processed,
            job_id=response.job_id
        )

    except Exception as e:
        logger.error(f"Batch processing error for job {job_id}: {str(e)}")
        supabase.table("jobs").update({
            "status": "error",
            "error_message": str(e),
            "completed_at": datetime.now(UTC).isoformat()
        }).eq("job_id", job_id).execute()
        raise HTTPException(status_code=500, detail=str(e))

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
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message="Limit reached. Please check usage in your account settings.",
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
            return QueryResponse(
                original_query=request_data.query,
                status="error",
                message="There was an error processing your request. This application may not have the ability to complete your request. You can also try rephrasing your request or breaking it down into multiple requests.",
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
                
                # Add cleanup task but DON'T execute immediately
                asyncio.create_task(temp_file_manager.cleanup_marked)
                
                # Update download URL to match client expectations
                download_url = f"/download?file_path={tmp_path}"
                
                supabase.table("jobs").update({
                    "status": "completed"
                }).eq("job_id", job_id).execute()

                return QueryResponse(
                    original_query=request_data.query,
                    status="completed",
                    message="Processing complete. Your file should download automatically.",
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
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message=f"Failed to process download: {str(e)[:100]}...",
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
                    "status": "completed"
                }).eq("job_id", job_id).execute()                

                # Only cleanup immediately for online type
                await temp_file_manager.cleanup_marked()
                await check_client_connection(request)
                if request_data.output_preferences.modify_existing:
                    message = f"Data successfully uploaded to {request_data.output_preferences.doc_name} - {request_data.output_preferences.sheet_name}."
                else:
                    message = f"Data successfully uploaded to new sheet in {request_data.output_preferences.doc_name}."

                return QueryResponse(
                    original_query=request_data.query,
                    status="completed",
                    message=message,
                    files=None,
                    num_images_processed=num_images_processed
                )
            except Exception as e:
                logger.error(f"Online destination upload failed for user {user_id}: {str(e)}")
                return QueryResponse(
                    original_query=request_data.query,
                    status="error", 
                    message=f"Failed to upload to destination.",
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
        
        return QueryResponse(
            original_query=request_data.query,
            status="error",
            message=error_msg,
            files=None,
            num_images_processed=0,
            error=str(e),
            total_pages=0
        )

@router.post("/process_query", response_model=QueryResponse)
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



