from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from typing import List, Optional, Annotated
from pydantic import BaseModel
from app.utils.sandbox import EnhancedPythonInterpreter
import json
import logging
from fastapi.responses import Response
from app.utils.preprocessing import preprocess_files
from app.utils.data_processing import get_data_snapshot
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.llm_service import LLMService
from app.utils.connection_status import check_client_connection
from app.utils.visualize_data import generate_visualization
from supabase.client import Client as SupabaseClient
from app.utils.file_management import temp_file_manager
from app.schemas import FileMetadata, InputUrl
import os
import base64

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class VisualizationOptions(BaseModel):
    color_palette: str
    custom_instructions: Optional[str] = None

class DataVisualizationRequest(BaseModel):
    files_metadata: Optional[List[FileMetadata]] = None
    input_urls: Optional[List[InputUrl]] = None
    options: VisualizationOptions

class VisualizationResponse(BaseModel):
    status: str = "success"
    image_data: str = None # base64 encoded image
    generated_image_name: str = None
    error: Optional[str] = None
    message: str = "Visualization generated successfully"

@router.post("/visualize_data", response_model=VisualizationResponse)
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
        logger.info(f"Request data: {request_data}")
        logger.info(f"Processing visualization for user {user_id}")
        
        # Initial connection check
        await check_client_connection(request)

        # Create temporary directory for session
        session_dir = temp_file_manager.get_temp_dir()

        # Preprocess files/urls to get dataframe
        preprocessed_data, _ = await preprocess_files(
            request=request,
            files=files,
            files_metadata=request_data.files_metadata,
            input_urls=request_data.input_urls,
            query=request_data.options.custom_instructions,
            session_dir=session_dir,
            supabase=supabase,
            user_id=user_id,
            llm_service=llm_service,
            num_images_processed=0
        )
        await check_client_connection(request)

        # Initialize sandbox with increased timeout for visualization
        sandbox = EnhancedPythonInterpreter(timeout_seconds=120)

        # Generate visualization
        buf = await generate_visualization(
            data=preprocessed_data,
            color_palette=request_data.options.color_palette,
            custom_instructions=request_data.options.custom_instructions,
            sandbox=sandbox,
            llm_service=llm_service,
            request=request
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

        # Save the image to a temporary file
        temp_image_path = os.path.join(session_dir, generated_image_name)
        with open(temp_image_path, "wb") as f:
            f.write(buf.getvalue())

        return VisualizationResponse(
            status="success",
            image_data=image_data,
            generated_image_name=generated_image_name,
            message="Visualization generated successfully"
        )

    except ValueError as e:
        logger.error(f"Visualization error for user {user_id}: {str(e)}")
        return VisualizationResponse(
            status="error",
            error=str(e),
            message="Error generating visualization",
            image_data=None
        )
    except Exception as e:
        error_msg = str(e)[:200] if str(e).isascii() else f"Error processing request: {e.__class__.__name__}"
        logger.error(f"Visualization error for user {user_id}: {error_msg}")
        return VisualizationResponse(
            status="error",
            error=error_msg,
            message="An unexpected error occurred while creating the visualization",
            image_data=None
        )
