from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends, Request
from typing import List, Optional, Annotated
from pydantic import BaseModel
from app.utils.sandbox import EnhancedPythonInterpreter
import json
import io
import logging
from fastapi.responses import Response
from app.utils.preprocessing import preprocess_files
from app.utils.data_processing import get_data_snapshot
from app.utils.auth import get_current_user, get_supabase_client
from app.utils.llm_service import get_llm_service, LLMService
from app.utils.check_connection import check_client_connection
from app.utils.visualize_data import generate_visualization
from supabase.client import Client as SupabaseClient
from app.utils.file_management import temp_file_manager
import matplotlib.pyplot as plt

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class DataVisualizationRequest(BaseModel):
    files_metadata: Optional[List[dict]] = None
    input_urls: Optional[List[str]] = None
    color_palette: str
    custom_instructions: Optional[str] = None

@router.post("/visualize_data")
async def create_visualization(
    request: Request,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    llm_service: LLMService = Depends(get_llm_service),
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

        # Preprocess files/urls to get dataframe
        preprocessed_data, _ = await preprocess_files(
            request=request,
            files=files,
            files_metadata=request_data.files_metadata,
            input_urls=request_data.input_urls,
            query=request_data.custom_instructions,
            session_dir=session_dir,
            supabase=supabase,
            user_id=user_id,
            llm_service=llm_service,
            num_images_processed=0
        )

        # Initialize sandbox with increased timeout for visualization
        sandbox = EnhancedPythonInterpreter(timeout_seconds=120)  # Increased timeout for complex plots

        # Generate visualization
        buf = await generate_visualization(
            data=preprocessed_data,
            color_palette=request_data.color_palette,
            custom_instructions=request_data.custom_instructions,
            sandbox=sandbox,
            llm_service=llm_service
        )

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={
                "Content-Disposition": "inline; filename=visualization.png"
            }
        )

    except ValueError as e:
        logger.error(f"Visualization error for user {user_id}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = str(e)[:200] if str(e).isascii() else f"Error processing request: {e.__class__.__name__}"
        logger.error(f"Visualization error for user {user_id}: {error_msg}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while creating the visualization."
        )
