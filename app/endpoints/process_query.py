from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Response
from typing import List, Optional, Any, Union
from pydantic import BaseModel
from app.utils.file_preprocessing import FilePreprocessor
from app.utils.process_query import process_query
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import FileDataInfo, QueryResponse, FileInfo, FileMetadata
import json
from app.utils.file_management import temp_file_manager
from app.utils.vision_processing import VisionProcessor
import os
import logging
from app.schemas import QueryRequest
from fastapi.responses import FileResponse
import pandas as pd
from app.utils.document_integrations import DocumentIntegrations
from app.utils.file_postprocessing import create_pdf, create_xlsx, create_docx, create_txt, create_csv
from fastapi import BackgroundTasks
import magic  # Add this import at the top

router = APIRouter()

def _create_return_value_snapshot(self) -> str:
    """Creates a string representation of the return value"""
    try:
        if isinstance(self.return_value, tuple):
            return ', '.join(str(item) for item in self.return_value)
        return str(self.return_value)
    except Exception:
        return "<unprintable value>"

def get_data_snapshot(content: Any, data_type: str) -> str:
    """Generate appropriate snapshot based on data type"""
    if data_type == "DataFrame":
        return content.head(10).to_string()
    elif data_type == "json":
        # For JSON, return first few key-value pairs or array elements
        if isinstance(content, dict):
            snapshot_dict = dict(list(content.items())[:5])
            return json.dumps(snapshot_dict, indent=2)
        elif isinstance(content, list):
            return json.dumps(content[:5], indent=2)
        return str(content)[:500]
    elif data_type == "text":
        # Return first 500 characters for text
        return content[:500] + ("..." if len(content) > 500 else "")
    elif data_type == "image":
        content.file.seek(0)
        size = len(content.file.read())
        content.file.seek(0)
        return f"Image file: {content.filename}, Size: {size} bytes"
    return str(content)[:500]

async def preprocess_files(files: List[UploadFile], web_urls: List[str], query: str, session_dir) -> List[FileDataInfo]:
    """Helper function to preprocess files and web URLs"""
    preprocessor = FilePreprocessor()
    processed_data = []
    mime = magic.Magic(mime=True)
    
    # Process web URLs if provided
    for url in web_urls:
        try:
            logging.info(f"Processing URL: {url}")
            df = preprocessor.process_web_url(url)
            processed_data.append(
                FileDataInfo(
                    content=df,
                    snapshot=get_data_snapshot(df, "DataFrame"),
                    data_type="DataFrame",
                    original_file_name=url.split('/')[-1],
                    url=url
                )
            )
        except Exception as e:
            logging.error(f"Error processing URL {url}: {str(e)}")
            raise Exception(f"Error processing URL {url}: {str(e)}")
    
    # Process uploaded files
    if files:
        for file in files:
            try:
                # Read the first few bytes to determine the file type
                file_bytes = await file.read(1024)
                file.seek(0)  # Reset file pointer
                
                detected_mime_type = mime.from_buffer(file_bytes)
                file_ext = file.filename.split('.')[-1].lower()
                
                logging.info(f"Processing file: {file.filename}")
                logging.info(f"Reported MIME type: {file.content_type}")
                logging.info(f"Detected MIME type: {detected_mime_type}")
                logging.info(f"File extension: {file_ext}")
                
                # Process based on MIME type
                if detected_mime_type in ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                        'application/vnd.ms-excel',
                                        'text/csv',
                                        'application/csv']:
                    file.file.seek(0)
                    if 'sheet' in detected_mime_type or 'excel' in detected_mime_type:
                        df = preprocessor.process_excel(file.file)
                    else:
                        df = preprocessor.process_csv(file.file)
                    processed_data.append(
                        FileDataInfo(
                            content=df,
                            snapshot=get_data_snapshot(df, "DataFrame"),
                            data_type="DataFrame",
                            original_file_name=file.filename
                        )
                    )
                
                elif detected_mime_type in ['application/json', 'text/json']:
                    file.file.seek(0)
                    json_content = json.loads(preprocessor.process_json(file.file))
                    processed_data.append(
                        FileDataInfo(
                            content=json_content,
                            snapshot=get_data_snapshot(json_content, "json"),
                            data_type="json",
                            original_file_name=file.filename
                        )
                    )
                
                elif detected_mime_type in ['text/plain', 
                                          'application/vnd.openxmlformats-officedocument.wordprocessingml.document']:
                    file.file.seek(0)
                    if 'wordprocessingml' in detected_mime_type:
                        content = preprocessor.process_docx(file.file)
                    else:
                        content = preprocessor.process_text(file.file)
                    processed_data.append(
                        FileDataInfo(
                            content=content,
                            snapshot=get_data_snapshot(content, "text"),
                            data_type="text",
                            original_file_name=file.filename
                        )
                    )
                
                elif detected_mime_type.startswith('image/'):
                    file.file.seek(0)
                    new_path = None
                    if detected_mime_type == 'image/png':
                        new_path = preprocessor.process_image(
                            file.file,
                            output_path=str(session_dir / f"{file.filename}.jpeg")
                        )
                    
                    vision_processor = VisionProcessor()
                    image_path = new_path or str(session_dir / file.filename)
                    
                    # Save the original file if it wasn't a PNG
                    if not new_path:
                        with open(image_path, 'wb') as f:
                            file.file.seek(0)
                            f.write(file.file.read())
                    
                    vision_result = vision_processor.process_image_with_vision(
                        image_path=image_path,
                        query=query
                    )
                    
                    if vision_result["status"] == "error":
                        raise Exception(f"Vision API error: {vision_result['error']}")
                    
                    processed_data.append(
                        FileDataInfo(
                            content=vision_result["content"],
                            snapshot=get_data_snapshot(file, "image"),
                            data_type="image",
                            original_file_name=file.filename,
                            new_file_path=new_path
                        )
                    )

                    # Clean up temporary files after vision processing is complete
                    if new_path:
                        try:
                            os.remove(new_path)
                        except Exception as e:
                            logging.warning(f"Failed to remove temporary file {new_path}: {e}")
                    try:
                        os.remove(image_path)
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary file {image_path}: {e}")
                
                elif detected_mime_type == 'application/pdf':
                    content, data_type, is_readable = preprocessor.process_pdf(file.file, query)
                    processed_data.append(
                        FileDataInfo(
                            content=content,
                            snapshot=get_data_snapshot(content, "text"),
                            data_type=data_type,
                            original_file_name=file.filename,
                            metadata={"is_readable": is_readable}
                        )
                    )
                else:
                    logging.warning(f"Unsupported MIME type: {detected_mime_type} for file {file.filename}")
                    raise Exception(f"Unsupported file type: {detected_mime_type}")

            except Exception as e:
                logging.error(f"Error processing file {file.filename}: {str(e)}")
                raise Exception(f"Error processing file {file.filename}: {str(e)}")
    
    return processed_data

async def handle_destination_upload(data: Any, destination_url: str) -> bool:
    """Upload data to various destination types"""
    try:
        doc_integrations = DocumentIntegrations()
        url_lower = destination_url.lower()
        
        if "docs.google.com" in url_lower:
            if "document" in url_lower:
                return await doc_integrations.append_to_google_doc(data, destination_url)
            elif "spreadsheets" in url_lower:
                return await doc_integrations.append_to_google_sheet(data, destination_url)
        
        elif "onedrive" in url_lower or "sharepoint.com" in url_lower:
            if "docx" in url_lower:
                return await doc_integrations.append_to_office_doc(data, destination_url)
            elif "xlsx" in url_lower:
                return await doc_integrations.append_to_office_sheet(data, destination_url)
        
        raise ValueError(f"Unsupported destination URL type: {destination_url}")
    
    except Exception as e:
        logging.error(f"Failed to upload to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/process_query", response_model=QueryResponse)
async def process_query_endpoint(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> QueryResponse:
    try:
        logging.info(f"Processing query with {len(request.files)} files")
        
        # Avoid logging the full request object
        logging.debug(f"Query: {request.query[:100]}...")  # Only log first 100 chars
        
        session_dir = temp_file_manager.get_temp_dir()
        
        try:
            preprocessed_data = await preprocess_files(
                files=request.files,
                web_urls=request.web_urls,
                query=request.query,
                session_dir=session_dir
            )
        except Exception as e:
            logging.error(f"File preprocessing error: {e.__class__.__name__}")
            return QueryResponse(
                status="error",
                message="Error processing uploaded files"
            )

        # Process the query with the processed data
        sandbox = EnhancedPythonInterpreter()
        result = process_query(
            query=request.query,
            sandbox=sandbox,
            data=preprocessed_data
        )
        result.return_value_snapshot = _create_return_value_snapshot(result.return_value)
        print("Query processed")
        print("Output preferences:", request.output_preferences.type)
        # Handle output based on type
        if request.output_preferences.type == "download":
            # Get the desired output format, defaulting based on data type
            output_format = request.output_preferences.format
            if not output_format:
                if isinstance(result.return_value, pd.DataFrame):
                    output_format = 'csv'
                elif isinstance(result.return_value, (dict, list)):
                    output_format = 'json'
                else:
                    output_format = 'txt'

            # Create temporary file in requested format
            if output_format == 'pdf':
                tmp_path = create_pdf(result.return_value)
                media_type = 'application/pdf'
            elif output_format == 'xlsx':
                tmp_path = create_xlsx(result.return_value)
                media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            elif output_format == 'docx':
                tmp_path = create_docx(result.return_value)
                media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            elif output_format == 'txt':
                tmp_path = create_txt(result.return_value)
                media_type = 'text/plain'
            else:  # csv
                tmp_path = create_csv(result.return_value)
                media_type = 'text/csv'

            # Mark files for cleanup after response is sent
            temp_file_manager.mark_for_cleanup(tmp_path, session_dir)
            background_tasks.add_task(temp_file_manager.cleanup_marked)
            
            return QueryResponse(
                status="success",
                message="File ready for download",
                files=[FileInfo(
                    file_path=str(tmp_path),
                    media_type=media_type,
                    filename=f'query_results.{output_format}',
                    download_url=f"/api/process_query/download/query_results.{output_format}"
                )]
            )

        elif request.output_preferences.type == "online":
            if not request.output_preferences.destination_url:
                raise ValueError("destination_url is required for online type")
                
            # Handle destination URL upload
            await handle_destination_upload(
                result.return_value,
                request.output_preferences.destination_url
            )

            temp_file_manager.cleanup_marked()  # Clean up immediately

            return QueryResponse(
                status="success",
                message="Data successfully uploaded to destination"
            )
        
        else:
            raise ValueError(f"Invalid output type: {request.output_preferences.type}")


    except Exception as e:
        error_msg = "Error processing binary file" if isinstance(e, UnicodeDecodeError) else str(e)
        logging.error(f"Process query error: {e.__class__.__name__}")
        return QueryResponse(
            status="error",
            message=error_msg
        )

@router.get("/download/{filename}")
async def download_file(filename: str):
    """
    Serve a processed file for download
    
    Args:
        filename: Name of the file to download
    """
    try:
        # Get the base directory where temporary files are stored
        base_dir = temp_file_manager.base_dir
        
        # Search for the file in all session directories
        for session_dir in base_dir.glob("session_*"):
            file_path = session_dir / filename
            if file_path.exists():
                # Determine the media type based on file extension
                extension = filename.split('.')[-1].lower()
                media_types = {
                    'pdf': 'application/pdf',
                    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'txt': 'text/plain',
                    'csv': 'text/csv'
                }
                
                media_type = media_types.get(extension, 'application/octet-stream')
                
                # Return the file as a response
                return FileResponse(
                    path=str(file_path),
                    media_type=media_type,
                    filename=filename,
                    background=None  # Don't delete the file immediately after sending
                )
        
        raise HTTPException(status_code=404, detail="File not found")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error serving file: {str(e)}")
