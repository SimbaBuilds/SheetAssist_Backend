from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response, Depends
from typing import List, Optional, Any, Union, Annotated
from pydantic import BaseModel
from app.utils.file_preprocessing import FilePreprocessor
from app.utils.process_query import process_query
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import QueryResponse, FileInfo, TruncatedSandboxResult
import json
from app.utils.file_management import temp_file_manager
import os
import logging
from app.schemas import QueryRequest
from fastapi.responses import FileResponse
import pandas as pd
from app.utils.file_postprocessing import handle_destination_upload, handle_download
from app.utils.data_processing import get_data_snapshot
from app.utils.file_preprocessing import preprocess_files
from fastapi import BackgroundTasks

from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/process_query", response_model=QueryResponse)
async def process_query_endpoint(
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    json_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> QueryResponse:
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    try:
        # Initialize truncated_result as None at the start
        truncated_result = None
        
        request = QueryRequest(**json.loads(json_data))
        logger.info(f"Processing query for user {user_id} with {len(request.files_metadata or [])} files")
        
        session_dir = temp_file_manager.get_temp_dir()
        logger.info("Calling preprocess_files")
        try:
            preprocessed_data = preprocess_files(
                files=files,
                files_metadata=request.files_metadata,
                web_urls=request.web_urls,
                query=request.query,
                session_dir=session_dir
            )
        except Exception as e:
            raise ValueError(e)

        # Process the query with the processed data
        sandbox = EnhancedPythonInterpreter()
        result = process_query(
            query=request.query,
            sandbox=sandbox,
            data=preprocessed_data
        )
        
        result.return_value_snapshot = get_data_snapshot(result.return_value, type(result.return_value).__name__)
        logger.info(f"Query processed for user {user_id} with return value snapshot type: {type(result.return_value).__name__}")

        if result.error:
            raise HTTPException(status_code=400, detail=result.error + " -- please try rephrasing your request")
        
        logger.info(f"""Output preferences for user {user_id}: 
                    type={request.output_preferences.type}, 
                    format={request.output_preferences.format}, 
                    destination_url={request.output_preferences.destination_url}, 
                    modify_existing={request.output_preferences.modify_existing}
                    """)
        
        # Handle output based on type
        if request.output_preferences.type == "download":
            tmp_path, media_type = handle_download(result, request, preprocessed_data)
            # Add cleanup task but DON'T execute immediately
            background_tasks.add_task(temp_file_manager.cleanup_marked)
            
            # Update download URL to match client expectations
            download_url = f"/download?file_path={tmp_path}"
            
            truncated_result = TruncatedSandboxResult(
                original_query=result.original_query,
                print_output=result.print_output,
                error=result.error,
                timed_out=result.timed_out,
                return_value_snapshot=result.return_value_snapshot
            )
            return QueryResponse(
                result=truncated_result,
                status="success",
                message="File ready for download",
                files=[FileInfo(
                    file_path=str(tmp_path),
                    media_type=media_type,
                    filename=os.path.basename(tmp_path),
                    download_url=download_url
                )]
            )

        elif request.output_preferences.type == "online":
            if not request.output_preferences.destination_url:
                raise ValueError("destination_url is required for online type")
                
            # Handle destination URL upload
            await handle_destination_upload(
                result.return_value,
                request,
                preprocessed_data,
                supabase,
                user_id
            )

            # Create truncated result before returning
            truncated_result = TruncatedSandboxResult(
                original_query=result.original_query,
                print_output=result.print_output,
                error=result.error,
                timed_out=result.timed_out,
                return_value_snapshot=result.return_value_snapshot
            )

            # Only cleanup immediately for online type
            temp_file_manager.cleanup_marked()

            return QueryResponse(
                result=truncated_result,
                status="success",
                message="Data successfully uploaded to destination",
                files=None
            )
        
        else:
            raise ValueError(f"Invalid output type: {request.output_preferences.type}")

    except Exception as e:
        # Enhanced error handling for binary content
        try:
            error_msg = str(e)
            if not error_msg.isascii() or len(error_msg) > 200:
                error_msg = f"Error processing request: {e.__class__.__name__}"
            else:
                error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
        except:
            error_msg = "An unexpected error occurred"
            
        logger.error(f"Process query error for user {user_id}: {error_msg}")
        
        # Create an error truncated_result if it wasn't created in the try block
        if truncated_result is None:
            truncated_result = TruncatedSandboxResult(
                original_query=request.query,
                print_output="",
                error=error_msg,
                timed_out=False,
                return_value_snapshot=""
            )
            
        return QueryResponse(
            result=truncated_result,
            status="error",
            message="An error occurred while processing your request -- please try again or rephrase your request",
            files=None
        )