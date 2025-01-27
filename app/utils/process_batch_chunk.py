from fastapi import UploadFile, HTTPException
from typing import List, Optional
from app.utils.process_query_algo import process_query_algo
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import BatchProcessingFileInfo, ChunkResponse, FileDataInfo
from app.utils.s3_file_management import temp_file_manager
import logging
from app.schemas import QueryRequest
from app.utils.postprocessing import handle_batch_chunk_result
from app.utils.data_processing import get_data_snapshot
from app.utils.preprocessing import preprocess_files
from supabase.client import Client as SupabaseClient
from datetime import datetime, UTC
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_batch_chunk(
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
            "started_at": datetime.now(UTC).isoformat()
        }).eq("job_id", job_id).execute()

        # Create new UploadFile objects for preprocessing
        upload_files = []
        for file in files:
            if hasattr(file, 'file'):
                content = await file.read()
                await file.seek(0)
            else:
                content = file.read()
                file.seek(0)
            
            file_obj = io.BytesIO(content)
            upload_file = UploadFile(
                file=file_obj,
                filename=file.filename if hasattr(file, 'filename') else 'file',
                headers={'content-type': file.content_type if hasattr(file, 'content_type') else 'application/octet-stream'}
            )
            upload_files.append(upload_file)

        # PRE-PROCESS
        try:
            preprocessed_data, num_images_processed = await preprocess_files(
                files=upload_files,
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

        #append PREVIOUS chunk to input data if it exists and output type is DOWNLOAD (online sheet output appends each batch, persisting past results automatically)
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

        #PROCESS_QUERY_ALGO
        result = await process_query_algo( 
            request=None,  # Not needed for core processing
            query=request_data.query,
            sandbox=sandbox,
            data=preprocessed_data,
            batch_context={"current": current_chunk + 1, "total": len(page_chunks)},
            contains_image_or_like=contains_image_or_like
        )
        
        #Update job status and chunk status
        error_msg = f"Chunk processing error: {result.error}" if result.error else None
        chunk_status = job_data["chunk_status"]
        if result.error:
            chunk_status.append(f"Chunk {current_chunk+1}: Error")
        else:
            chunk_status.append(f"Chunk {current_chunk+1}: Success") 

        
        supabase.table("batch_jobs").update({
            "status": "processing",
            "error_message": error_msg,
            "chunk_status": chunk_status

        }).eq("job_id", job_id).execute()
        job_response = supabase.table("batch_jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()

        logging.info(f"CHUNK STATUS: {job_response.data[0]['chunk_status']}")

        

            
        # Handle post-processing
        try:
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
        except Exception as e:
            logger.error(f"Error in handle_batch_chunk_result for chunk {current_chunk}: {str(e)}")
            chunk_status = job_data["chunk_status"]
            new_chunk_status = chunk_status + f"Chunk {current_chunk+1}: Error"
            supabase.table("batch_jobs").update({
                "status": "processing",
                "error_message": f"Error processing chunk {current_chunk+1}: {str(e)}",
                "chunk_status": new_chunk_status
            }).eq("job_id", job_id).execute()

        # Cleanup temporary files
        temp_file_manager.cleanup_marked()

        # Update job with images processed count
        supabase.table("batch_jobs").update({
            "images_processed": num_images_processed,
            "status": "processing",
        }).eq("job_id", job_id).execute()

        return ChunkResponse(
            result=result,
            status=status,
            message=f"Batch {current_chunk + 1}/{len(page_chunks)} processed successfully",
            files=None,
            num_images_processed=num_images_processed,
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