from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from typing import List, Optional, Annotated, Union
from pydantic import BaseModel
from app.utils.sandbox import EnhancedPythonInterpreter
import json
import logging
from fastapi.responses import Response
from app.utils.preprocessing import preprocess_files
from app.utils.data_processing import get_data_snapshot
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.llm_service import LLMService
from app.utils.connection_and_status import check_client_connection
from app.utils.visualize_data import generate_visualization
from supabase.client import Client as SupabaseClient
from app.utils.s3_file_management import temp_file_manager
from app.utils.s3_file_actions import s3_file_actions
from app.schemas import FileUploadMetadata, InputUrl
import os
import base64
import io
import asyncio

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class VisualizationOptions(BaseModel):
    color_palette: str
    custom_instructions: Optional[str] = None

class DataVisualizationRequest(BaseModel):
    files_metadata: Optional[List[FileUploadMetadata]] = None
    input_urls: Optional[List[InputUrl]] = None
    options: VisualizationOptions

class VisualizationSuccessResponse(BaseModel):
    success: bool = True
    image_data: str = None # base64 encoded image
    generated_image_name: str = None
    message: str = "Visualization generated successfully"
    s3_key: Optional[str] = None
    s3_url: Optional[str] = None

class VisualizationErrorResponse(BaseModel):
    success: bool = False
    error: str = None
    message: str = "Error generating visualization"

@router.post("/visualize_data", response_model=Union[VisualizationSuccessResponse, VisualizationErrorResponse])
async def create_visualization(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    llm_service: LLMService = Depends(LLMService),
    json_data: str = Form(...),
    files: List[UploadFile] = File(default=[]),
):
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")

    try:
        # Parse request data
        request_data = DataVisualizationRequest(**json.loads(json_data))
        logger.info(f"Processing visualization for user {user_id}")
        
        # Initial connection check
        await check_client_connection(request)

        # Create temporary directory for session
        session_dir = temp_file_manager.get_temp_dir()

        # Store file contents in memory
        files_data = []
        for i, file_meta in enumerate(request_data.files_metadata or []):
            try:
                if file_meta.s3_key:
                    # Handle S3 files
                    stream = await s3_file_actions.get_streaming_body(file_meta.s3_key)
                    content = await asyncio.to_thread(stream.read)
                    files_data.append({
                        'content': content,
                        'filename': file_meta.name,
                        'content_type': file_meta.type
                    })
                    await asyncio.to_thread(stream.close)
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
                return VisualizationErrorResponse(
                    success=False,
                    error=str(e),
                    message=f"Failed to read file {file_meta.name}"
                )

        # Create new UploadFile objects for preprocessing
        upload_files = []
        for file_data in files_data:
            file_obj = io.BytesIO(file_data['content'])
            upload_file = UploadFile(
                file=file_obj,
                filename=file_data['filename'],
                headers={'content-type': file_data['content_type']}
            )
            upload_files.append(upload_file)

        try:
            # Preprocess files/urls to get dataframe
            preprocessed_data, _ = await preprocess_files(
                files=upload_files,
                files_metadata=request_data.files_metadata,
                input_urls=request_data.input_urls,
                query=request_data.options.custom_instructions,
                session_dir=session_dir,
                supabase=supabase,
                user_id=user_id,
                num_images_processed=0
            )
        except ValueError as preprocess_error:
            logger.error(f"Preprocessing error: {str(preprocess_error)}")
            return VisualizationErrorResponse(
                success=False,
                error=str(preprocess_error),
                message="Failed to preprocess input files"
            )
            
        await check_client_connection(request)

        # Initialize sandbox with increased timeout for visualization
        sandbox = EnhancedPythonInterpreter(timeout_seconds=120)

        try:
            # Generate visualization
            buf = await generate_visualization(
                data=preprocessed_data,
                color_palette=request_data.options.color_palette,
                custom_instructions=request_data.options.custom_instructions,
                sandbox=sandbox,
                llm_service=llm_service,
                request=request
            )
        except Exception as viz_error:
            logger.error(f"Visualization generation error: {str(viz_error)}")
            return VisualizationErrorResponse(
                success=False,
                error=str(viz_error),
                message="Failed to generate visualization"
            )

        await check_client_connection(request)

        # Get the buffer contents and encode as base64
        image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Generate a relevant filename using the LLM service
        generated_image_name = await llm_service.execute_with_fallback(
            "file_namer",
            request_data.options.custom_instructions or "visualization",
            preprocessed_data
        )
        generated_image_name = f"{generated_image_name[1]}.png"  # Add .png extension to the generated name

        # # Save to S3 if the image is large
        # s3_key = None
        # s3_url = None
        # if len(image_data) > 100 * 1024:  # If larger than 100KB
        #     try:
        #         s3_key = f"visualizations/{user_id}/{generated_image_name}"
        #         await s3_file_actions.stream_upload(io.BytesIO(buf.getvalue()), s3_key)
        #         s3_url = s3_file_actions.get_presigned_url(s3_key)
        #         image_data = None  # Don't send the base64 data if we have S3
        #     except Exception as s3_error:
        #         logger.error(f"S3 upload error: {str(s3_error)}")
        #         # Continue with base64 if S3 upload fails
        #         s3_key = None
        #         s3_url = None

        # # Save locally only if not in S3
        # if not s3_key:
        #     temp_image_path = os.path.join(session_dir, generated_image_name)
        #     with open(temp_image_path, "wb") as f:
        #         f.write(buf.getvalue())

        return VisualizationSuccessResponse(
            success=True,
            image_data=image_data,
            generated_image_name=generated_image_name,
            message="Visualization generated successfully",
        )

    except Exception as e:
        error_msg = str(e)[:200] if str(e).isascii() else f"Error processing request: {e.__class__.__name__}"
        logger.error(f"Visualization error for user {user_id}: {error_msg}")
        return VisualizationErrorResponse(
            success=False,
            error=error_msg,
            message="An unexpected error occurred while creating the visualization"
        )
