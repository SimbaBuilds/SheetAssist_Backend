from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response, Depends, Request
from typing import List, Optional, Any, Union, Annotated
from pydantic import BaseModel
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
from app.utils.postprocessing import handle_destination_upload, handle_download
from app.utils.data_processing import get_data_snapshot
from app.utils.preprocessing import preprocess_files
from fastapi import BackgroundTasks
from dotenv import load_dotenv
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.llm_service import get_llm_service, LLMService
from app.utils.check_connection import check_client_connection
# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()




@router.post("/process_query", response_model=QueryResponse)
async def process_query_endpoint(
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