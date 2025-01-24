from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request, Response
from typing import List, Annotated, Optional
from app.utils.process_query_algo import process_query_algo
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import QueryResponse, FileInfo, TruncatedSandboxResult, BatchProcessingFileInfo, ChunkResponse, FileDataInfo
import json
from app.utils.s3_file_management import temp_file_manager
import os
import logging
from app.schemas import QueryRequest
from app.utils.postprocessing import handle_destination_upload, handle_download, handle_batch_chunk_result
from app.utils.data_processing import get_data_snapshot
from app.utils.preprocessing import preprocess_files, determine_pdf_page_count, pdf_classifier, check_limits_pre_batch
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.connection_and_status import check_client_connection, construct_status_response
from app.schemas import InputUrl
from datetime import datetime, UTC
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import io
# Add at the top of the file, after imports


load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()



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
        request_data = QueryRequest(**json.loads(json_data))
        
        # Store file contents in memory before background processing
        files_data = []
        for file in files:
            content = await file.read()
            await file.seek(0)  # Reset for immediate use if needed
            files_data.append({
                'content': content,
                'filename': file.filename,
                'content_type': file.content_type
            })
        
        total_pages = 0
        # Update page counts using the original files
        if request_data.files_metadata:
            for file_meta, upload_file in zip(request_data.files_metadata, files):
                if not file_meta.page_count:
                    page_count = await determine_pdf_page_count(upload_file)
                    file_meta.page_count = page_count
                    total_pages += file_meta.page_count
                    await upload_file.seek(0)

        #If destination url and no input url, add destination url and sheet name to input urls for processing context
        if request_data.output_preferences.destination_url and not request_data.input_urls and request_data.output_preferences.modify_existing:
            request_data.input_urls = [InputUrl(url=request_data.output_preferences.destination_url, sheet_name=request_data.output_preferences.sheet_name)]
        
        # Check if any PDF has more than CHUNK_SIZE pages
        need_to_batch = False
        contains_image_or_like = False 
        
        i = 0
        # Check for images to propagate flag to llm service for prompt selection
        for file_meta in (request_data.files_metadata or []):
            if files_data[i]['content_type'] == "image/jpeg" or files_data[i]['content_type'] == "image/png" or files_data[i]['content_type'] == "image/jpg":
                contains_image_or_like = True
            if files_data[i]['content_type'] == "application/pdf":
                contains_image_or_like = pdf_classifier(files_data[i]['content'])

        i = 0
        for file_meta in (request_data.files_metadata or []):
            logger.info(f"file_meta: {file_meta.page_count}")
            if file_meta.page_count and file_meta.page_count > int(os.getenv("CHUNK_SIZE")):
                if files_data[i]['content_type'] == "application/pdf":
                    is_image_like_pdf = pdf_classifier(files_data[i]['content'])
                    if is_image_like_pdf: 
                        contains_image_or_like = True
                    if is_image_like_pdf: #and greater than CHUNK_SIZE
                        need_to_batch = True 
                        logger.info(f"----- Need_to_batch: {need_to_batch} -----")
                        break
            i += 1
        
        logger.info(f"(01/19) contains_image_or_like: {contains_image_or_like}")
        # Decision routing based on page count
        if need_to_batch:
            # Generate unique job ID
            logger.info(f"Large PDF detected")
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
            job_id = f"job_{user_id}_{int(time.time())}"
            
            # Calculate page ranges for each file
            page_chunks = []
            CHUNK_SIZE = int(os.getenv("CHUNK_SIZE")) 
            logger.info(f"CHUNK_SIZE: {CHUNK_SIZE}")
            
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
                "output_preferences": request_data.output_preferences.model_dump(),
                "created_at": datetime.now(UTC).isoformat(),
                "started_at": None,
                "completed_at": None,
                "error_message": None,
                "result_snapshot": None,
                "result_file_path": None,
                "result_media_type": None,
                "page_chunks": page_chunks,
                "current_chunk": 0,
                "query": request_data.query,
            }).execute()
            
            # Add logging before background task
            logger.info(f"Before background task - Files state: {[f.file._file.closed for f in files]}")
            
            # Create new UploadFile objects for the background task
            batch_files = []
            for file in files_data:
                file_obj = io.BytesIO(file['content'])
                upload_file = UploadFile(
                    file=file_obj,
                    filename=file['filename'],
                    headers={'content-type': file['content_type']}
                )
                batch_files.append(upload_file)

            # Start first batch processing in background
            background_tasks.add_task(
                process_query_batch_endpoint,
                request=request,
                user_id=user_id,
                supabase=supabase,
                request_data=request_data,
                files=batch_files,  # Use the new file objects
                job_id=job_id,
                contains_image_or_like=contains_image_or_like
            )
            
            return QueryResponse(
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
                request_data=request_data,
                files=files,
                background_tasks=background_tasks,
                contains_image_or_like=contains_image_or_like
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
        job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = job_response.data[0]
        
        # Construct response based on job status
        return construct_status_response(job)
            
    except Exception as e:
        logger.error(f"Error in process_query_status_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process_query_standard", response_model=QueryResponse)
async def process_query_standard_endpoint(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    request_data: QueryRequest,
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    contains_image_or_like: bool = False
) -> QueryResponse:



    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    num_images_processed = 0

    logger.info( f"process_query_standard_endpoint called")


    try:
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
                files=files,
                files_metadata=request_data.files_metadata,
                input_urls=request_data.input_urls,
                query=request_data.query,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                num_images_processed=num_images_processed
            )
        except ValueError as e:
            # Specifically handle image/overage limit errors
            error_msg = str(e)
            if "limit reached" in error_msg.lower():
                return QueryResponse(
                    original_query=request_data.query,
                    status="error",
                    message = "Limit reached.  Please check usage in your account settings.",
                    files=None,
                    num_images_processed=0,
                    error=error_msg,
                    total_pages=0
                )
            raise  # Re-raise other ValueError exceptions


        except Exception as e:
            raise ValueError(e)

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
                message="There was an error processing your request.  This application may not have the ability to complete your request.  You can also try rephrasing your request or breaking it down into multiple requests.",
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
                background_tasks.add_task(temp_file_manager.cleanup_marked)
                
                # Update download URL to match client expectations
                download_url = f"/download?file_path={tmp_path}"
                
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

                # Only cleanup immediately for online type
                temp_file_manager.cleanup_marked()
                await check_client_connection(request)
                if request_data.output_preferences.modify_existing:
                    message = f"Data successfully uploaded to  {request_data.output_preferences.doc_name} - {request_data.output_preferences.sheet_name}."
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

async def _process_batch_chunk(
    user_id: str,
    supabase: SupabaseClient,
    request_data: QueryRequest,
    files: List[UploadFile],
    job_id: str,
    session_dir: str,
    current_chunk: Optional[int] = None,
    previous_chunk_return_value: Optional[tuple] = None,
    contains_image_or_like: bool = False
) -> ChunkResponse:
    """Internal function to process a single batch chunk without route dependencies."""
    try:
        # Get job data and validate chunk
        logger.info(f"\n\n------- Processing chunk: {current_chunk + 1} (Index: {current_chunk}) -----------\n")
        
        job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).execute()
        if not job_response.data or len(job_response.data) == 0:
            raise ValueError("Job not found")
            
        job_data = job_response.data[0]
        current_chunk = current_chunk if current_chunk is not None else job_data["current_chunk"]
        page_chunks = job_data["page_chunks"]
        
        if current_chunk < 0 or current_chunk >= len(page_chunks):
            raise ValueError(f"Invalid chunk index: {current_chunk}. Valid range: 0-{len(page_chunks)-1}")
            
        current_processing_info = BatchProcessingFileInfo(**page_chunks[current_chunk])
        
        # Update status
        current_chunk_info = page_chunks[current_chunk]
        page_range = current_chunk_info['page_range']
        supabase.table("batch_jobs").update({
            "status": "processing",
            "started_at": datetime.now(UTC).isoformat(),
            "message": f"Processing pages {page_range[0] + 1} to {page_range[1]}"
        }).eq("job_id", job_id).execute()

        # Process chunk
        try:
            preprocessed_data, num_images_processed = await preprocess_files(
                files=files,
                files_metadata=[current_processing_info.metadata],
                input_urls=request_data.input_urls,
                query=request_data.query,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                num_images_processed=0,  
                page_range=current_processing_info.page_range,
            )
        except ValueError as e:
            error_msg = str(e)
            if "limit reached" in error_msg.lower():
                # Get current job data
                job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).execute()
                if job_response.data:
                    job_data = job_response.data[0]
                    page_chunks = job_data["page_chunks"]
                    
                    # Remove unprocessed chunks
                    updated_chunks = page_chunks[:current_chunk + 1]
                    
                    # Update job with error and truncated chunks
                    supabase.table("batch_jobs").update({
                        "status": "error",
                        "message": "Limit reached.  Please check usage in your account settings.",
                        "error_message": error_msg,
                        "page_chunks": updated_chunks,
                        "completed_at": datetime.now(UTC).isoformat()
                    }).eq("job_id", job_id).execute()

            raise ValueError(error_msg)

        #append previous chunk to input data if it exists and output type is download (online sheet output appends each batch, persisting past results automatically)
        if previous_chunk_return_value and request_data.output_preferences.type == "download":
            job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
            if not job_response.data:
                raise HTTPException(status_code=404, detail="Job not found")
            job = job_response.data[0]
            page_chunks = job.get('page_chunks', [])
            current_chunk = job.get('current_chunk', 0)
            file_name = page_chunks[current_chunk]['file_id']
            previous_chunk_df = previous_chunk_return_value[0]
            previous_chunk_data = FileDataInfo(
                content=previous_chunk_df,
                snapshot=get_data_snapshot(previous_chunk_df, "DataFrame"),
                data_type="DataFrame",
                original_file_name=file_name
            )
            preprocessed_data.append(previous_chunk_data)

        sandbox = EnhancedPythonInterpreter()
        #Obtain SandboxResult
        result = await process_query_algo( 
            request=None,  # Not needed for core processing
            query=request_data.query,
            sandbox=sandbox,
            data=preprocessed_data,
            batch_context={"current": current_chunk + 1, "total": len(page_chunks)},
            contains_image_or_like=contains_image_or_like
        )
        
        if result.error:
            error_msg = f"Chunk processing error: {result.error}"
            supabase.table("batch_jobs").update({
                "status": "error",
                "message": "There was an error processing your request.  This application may not have the ability to complete your request.  You can also try rephrasing your request or breaking it down into multiple requests",
                "error_message": error_msg,
                "completed_at": datetime.now(UTC).isoformat()
            }).eq("job_id", job_id).execute()
            
            raise ValueError(error_msg)

        # Handle post-processing
        status, result_file_path, result_media_type = await handle_batch_chunk_result(
            result=result,
            request_data=request_data,
            preprocessed_data=preprocessed_data,
            supabase=supabase,
            user_id=user_id,
            job_id=job_id,
            session_dir=session_dir,
            current_chunk=current_chunk,
            total_chunks=len(page_chunks),
            num_images_processed=num_images_processed,
        )

        # Cleanup temporary files
        temp_file_manager.cleanup_marked()

        # Update job with images processed count
        supabase.table("batch_jobs").update({
            "images_processed": num_images_processed,
            "status": "processing",
            "started_at": datetime.now(UTC).isoformat(),
            "message": f"Processing pages {page_range[0] + 1} to {page_range[1]}"
        }).eq("job_id", job_id).execute()

        return ChunkResponse(
            result=result,
            status=status,
            message=f"Batch {current_chunk + 1}/{len(page_chunks)} processed successfully",
            files=None,
            num_images_processed=num_images_processed,  # Include in response
            job_id=job_id
        )

    except ValueError as e:  # Specific handling for preprocessing errors
        logger.error(f"Processing error in batch chunk {current_chunk}: {str(e)}")
        supabase.table("batch_jobs").update({
            "status": "error",
            "error_message": str(e),
            "completed_at": datetime.now(UTC).isoformat()
        }).eq("job_id", job_id).execute()
        raise
    except Exception as e:
        logger.error(f"Error processing batch chunk {current_chunk}: {str(e)}")
        supabase.table("batch_jobs").update({
            "status": "error",
            "error_message": str(e),
            "completed_at": datetime.now(UTC).isoformat()
        }).eq("job_id", job_id).execute()
        raise


@router.post("/process_query/batch", response_model=QueryResponse)
async def process_query_batch_endpoint(
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

        # Store file contents for processing
        files_data = []
        for file in files:
            content = await file.read()
            await file.seek(0)
            files_data.append({
                'content': content,
                'filename': file.filename,
                'content_type': file.content_type
            })
        
        # Get job data
        job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).execute()
        if not job_response.data or len(job_response.data) == 0:
            raise ValueError("Job not found")
            
        job_data = job_response.data[0]
        page_chunks = job_data["page_chunks"]
        response = None

        # Process all chunks sequentially
        for chunk_index in range(len(page_chunks)):
            await check_client_connection(request)
            logger.info(f"Processing chunk {chunk_index + 1} of {len(page_chunks)}")
            
            # Create new file objects for each chunk
            chunk_files = []
            for file_data in files_data:
                file_obj = io.BytesIO(file_data['content'])
                upload_file = UploadFile(
                    filename=file_data['filename'],
                    file=file_obj,
                    headers={'content-type': file_data['content_type']}
                )
                chunk_files.append(upload_file)

            # Process the chunk for ChunkResponse
            response = await _process_batch_chunk(
                user_id=user_id,
                supabase=supabase,
                request_data=request_data,
                files=chunk_files,
                job_id=job_id,
                session_dir=session_dir,
                current_chunk=chunk_index,
                previous_chunk_return_value=previous_chunk_return_value,
                contains_image_or_like=contains_image_or_like
            )

            # Store the return value for the next chunk
            previous_chunk_return_value = response.result.return_value #tuple

            # Accumulate total images processed
            total_images_processed += response.num_images_processed

            # Update progress including images processed
            processed_pages = min((chunk_index + 1) * int(os.getenv("CHUNK_SIZE")), job_data["total_pages"])
            supabase.table("batch_jobs").update({
                "current_chunk": min(chunk_index + 1, len(page_chunks) - 1),
                "processed_pages": processed_pages,
                "total_images_processed": total_images_processed,
                "status": "processing" if chunk_index < len(page_chunks) - 1 else "completed",
                "completed_at": datetime.now(UTC).isoformat() if chunk_index == len(page_chunks) - 1 else None
            }).eq("job_id", job_id).execute()
        
        return QueryResponse(
            original_query=request_data.query,
            status=response.status,
            message=response.message,
            files=response.files,
            num_images_processed=response.num_images_processed,
            job_id=response.job_id
        )

    except Exception as e:
        logger.error(f"Batch processing error for job {job_id}: {str(e)}")
        supabase.table("batch_jobs").update({
            "status": "error",
            "error_message": str(e),
            "completed_at": datetime.now(UTC).isoformat()
        }).eq("job_id", job_id).execute()
        raise HTTPException(status_code=500, detail=str(e))



