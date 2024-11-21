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
import io
from typing import Dict
router = APIRouter()
def sanitize_error_message(e: Exception) -> str:
    """Sanitize error messages to remove binary data"""
    return str(e).encode('ascii', 'ignore').decode('ascii')

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

async def preprocess_files(files: List[Dict[str, bytes]], files_metadata: List[FileMetadata], web_urls: List[str], query: str, session_dir) -> List[FileDataInfo]:
    """Helper function to preprocess files and web URLs"""
    preprocessor = FilePreprocessor()
    processed_data = []
    
    # Process web URLs if provided
    for url in web_urls:
        try:
            logging.info(f"Processing URL: {url}")
            content = preprocessor.process_web_url(url)
            data_type = "DataFrame" if isinstance(content, pd.DataFrame) else "text"
            processed_data.append(
                FileDataInfo(
                    content=content,
                    snapshot=get_data_snapshot(content, data_type),
                    data_type=data_type,
                    original_file_name=url.split('/')[-1],
                    url=url
                )
            )
        except Exception as e:
            error_msg = sanitize_error_message(e)
            logging.error(f"Error processing URL {url}: {error_msg}")
            raise Exception(f"Error processing URL {url}: {error_msg}")
    
    # Process uploaded files using metadata
    if files and files_metadata:
        for metadata in files_metadata:
            try:
                # Find corresponding file in the list of dictionaries
                file_dict = next(
                    (f for f in files if f.get(f"file_{metadata.index}")), 
                    None
                )
                
                if not file_dict:
                    raise ValueError(f"No file content found for file: {metadata.name}")
                
                # Get the bytes content directly
                file_content = file_dict[f"file_{metadata.index}"]
                
                logging.info(f"Processing file: {metadata.name} with type: {metadata.type}")
                
                # Create BytesIO object from file content
                file_io = io.BytesIO(file_content)
                
                try:
                    # Process based on exact MIME types
                    match metadata.type:
                        case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' | 'text/csv':
                            if metadata.type.endswith('sheet'):
                                excel_file = io.BytesIO(file_content)
                                df = pd.read_excel(excel_file)
                            else:
                                df = preprocessor.process_csv(file_io)
                                
                            processed_data.append(
                                FileDataInfo(
                                    content=df,
                                    snapshot=get_data_snapshot(df, "DataFrame"),
                                    data_type="DataFrame",
                                    original_file_name=metadata.name
                                )
                            )

                        case 'image/png' | 'image/jpeg' | 'image/jpg':
                            # Create a copy of the file content
                            file_copy = io.BytesIO(file_content)
                            
                            new_path = preprocessor.process_image(
                                file_copy,
                                output_path=str(session_dir / f"{metadata.name}.jpeg")
                            )
                            
                            vision_processor = VisionProcessor()
                            image_path = new_path or str(session_dir / metadata.name)
                            
                            # Save the original file if it wasn't converted
                            if not new_path:
                                with open(image_path, 'wb') as f:
                                    file_copy.seek(0)
                                    f.write(file_copy.read())
                            
                            vision_result = vision_processor.process_image_with_vision(
                                image_path=image_path,
                                query=query
                            )
                            
                            if vision_result["status"] == "error":
                                raise ValueError(f"Vision API error: {vision_result['error']}")
                            
                            processed_data.append(
                                FileDataInfo(
                                    content=vision_result["content"],
                                    snapshot=get_data_snapshot(file_io, "image"),
                                    data_type="image",
                                    original_file_name=metadata.name,
                                    new_file_path=new_path
                                )
                            )
                        
                        case 'application/json':
                            json_content = preprocessor.process_json(file_io)
                            processed_data.append(
                                FileDataInfo(
                                    content=json_content,
                                    snapshot=get_data_snapshot(json_content, "json"),
                                    data_type="json",
                                    original_file_name=metadata.name
                                )
                            )
                        
                        case 'text/plain':
                            text_content = preprocessor.process_text(file_io)
                            processed_data.append(
                                FileDataInfo(
                                    content=text_content,
                                    snapshot=get_data_snapshot(text_content, "text"),
                                    data_type="text",
                                    original_file_name=metadata.name
                                )
                            )
                        
                        case 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                            doc_content = preprocessor.process_docx(file_io)
                            processed_data.append(
                                FileDataInfo(
                                    content=doc_content,
                                    snapshot=get_data_snapshot(doc_content, "text"),
                                    data_type="text",
                                    original_file_name=metadata.name
                                )
                            )
                        
                        case 'application/pdf':
                            content, data_type, is_readable = preprocessor.process_pdf(file_io, query)
                            processed_data.append(
                                FileDataInfo(
                                    content=content,
                                    snapshot=get_data_snapshot(content, "text"),
                                    data_type=data_type,
                                    original_file_name=metadata.name,
                                    metadata={"is_readable": is_readable}
                                )
                            )
                        
                        case _:
                            logging.warning(f"Unsupported MIME type: {metadata.type} for file {metadata.name}")
                            raise ValueError(f"Unsupported file type: {metadata.type}")

                except UnicodeDecodeError:
                    raise ValueError(f"Cannot process binary content in file: {metadata.name}")
                
                except Exception as e:
                    error_msg = sanitize_error_message(e)
                    raise ValueError(f"Error processing file {metadata.name}: {error_msg}")

            except Exception as e:
                error_msg = str(e).encode('ascii', 'ignore').decode('ascii')
                logging.error(f"Error processing file {metadata.name}: {error_msg}")
                raise ValueError(f"Error processing file {metadata.name}: {error_msg}")

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
    json: str = Form(...),
    files: List[Dict[str, UploadFile]] = File(None),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> QueryResponse:
    try:
        request = QueryRequest(**json.loads(json))
        logging.info(f"Processing query with {len(request.files_metadata or [])} files")
        session_dir = temp_file_manager.get_temp_dir()
        
        # Dictionary to store file contents
        file_contents: Dict[str, bytes] = {}
        
        # Process files if they exist
        if files:
            for file_dict in files:
                # Each file_dict should contain a single key-value pair
                for key, file in file_dict.items():
                    # Extract index from key (e.g., 'file_0' -> '0')
                    index = int(key.split('_')[1])
                    
                    # Find corresponding metadata
                    metadata = next((m for m in request.files_metadata if m.index == index), None)
                    if not metadata:
                        raise ValueError(f"No metadata found for file index: {index}")
                    
                    # Read and store file content
                    content = await file.read()
                    file_key = f"file_{index}"
                    file_contents[file_key] = content
                    
                    # Reset file pointer
                    await file.seek(0)
                    
                    logging.info(f"Processed file: {file.filename}, size: {len(content)} bytes")
        
        try:
            preprocessed_data = await preprocess_files(
                files=file_contents,
                files_metadata=request.files_metadata,
                web_urls=request.web_urls,
                query=request.query,
                session_dir=session_dir
            )
        except Exception as e:
            try:
                error_msg = str(e)
                if not error_msg.isascii() or len(error_msg) > 200:
                    error_msg = f"Error processing files: {e.__class__.__name__}"
                else:
                    error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
            except:
                error_msg = "Error processing files"
                
            logging.error(f"File preprocessing error: {e.__class__.__name__}")
            raise ValueError(error_msg)

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
        # Enhanced error handling for binary content
        try:
            error_msg = str(e)
            if not error_msg.isascii() or len(error_msg) > 200:
                error_msg = f"Error processing request: {e.__class__.__name__}"
            else:
                error_msg = error_msg.encode('ascii', 'ignore').decode('ascii')
        except:
            error_msg = "An unexpected error occurred"
            
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
