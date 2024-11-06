from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import List, Optional, Any
from pydantic import BaseModel
from app.utils.file_preprocessing import FilePreprocessor
from app.utils.process_query import process_query
from app.utils.sandbox import EnhancedPythonInterpreter
from app.schemas import FileDataInfo, ProcessedQueryResult
import json
from app.utils.file_management import temp_file_manager
from app.utils.vision_processing import VisionProcessor
import os
import logging
from app.schemas import QueryRequest
from fastapi.responses import FileResponse
import tempfile
import csv
import io
import pandas as pd
from app.utils.document_integrations import DocumentIntegrations
from app.utils.file_postprocessing import create_pdf

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
        # For images, return basic file info
        return f"Image file: {content.filename}, Size: {len(content.file.read())} bytes"
    return str(content)[:500]

async def preprocess_files(files: List[UploadFile], web_urls: List[str], query: str, session_dir) -> List[FileDataInfo]:
    """Helper function to preprocess files and web URLs"""
    preprocessor = FilePreprocessor()
    processed_data = []
    
    # Process web URLs if provided
    for url in web_urls:
        try:
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
            raise Exception(f"Error processing URL {url}: {str(e)}")
    
    # Process uploaded files
    if files:
        for file in files:
            file_ext = file.filename.split('.')[-1].lower()
            
            try:
                if file_ext in ['xlsx', 'xls', 'csv']:
                    df = preprocessor.process_excel(file.file) if file_ext in ['xlsx', 'xls'] else preprocessor.process_csv(file.file)
                    processed_data.append(
                        FileDataInfo(
                            content=df,
                            snapshot=get_data_snapshot(df, "DataFrame"),
                            data_type="DataFrame",
                            original_file_name=file.filename
                        )
                    )
                
                elif file_ext == 'json':
                    json_content = json.loads(preprocessor.process_json(file.file))
                    processed_data.append(
                        FileDataInfo(
                            content=json_content,
                            snapshot=get_data_snapshot(json_content, "json"),
                            data_type="json",
                            original_file_name=file.filename
                        )
                    )
                
                elif file_ext in ['txt', 'docx']:
                    content = preprocessor.process_docx(file.file) if file_ext == 'docx' else preprocessor.process_text(file.file)
                    processed_data.append(
                        FileDataInfo(
                            content=content,
                            snapshot=get_data_snapshot(content, "text"),
                            data_type="text",
                            original_file_name=file.filename
                        )
                    )
                
                elif file_ext in ['png', 'jpeg', 'jpg']:
                    new_path = None
                    if file_ext == 'png':
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
            
                elif file_ext == 'pdf':
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

            except Exception as e:
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
        
        elif "office.com" in url_lower or "sharepoint.com" in url_lower:
            if "word" in url_lower:
                return await doc_integrations.append_to_office_doc(data, destination_url)
            elif "excel" in url_lower:
                return await doc_integrations.append_to_office_sheet(data, destination_url)
        
        raise ValueError(f"Unsupported destination URL type: {destination_url}")
    
    except Exception as e:
        logging.error(f"Failed to upload to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def handle_output_preferences(result, output_preferences) -> Any:
    """Handle different output preferences and return appropriate response"""
    if not output_preferences:
        return {
            "message": "success",
            "result": (result.return_value.to_dict(orient='records') 
                    if hasattr(result.return_value, 'to_dict') 
                    else result.return_value)
        }

    if output_preferences.type == "download":
        # Create appropriate file format
        if isinstance(result.return_value, pd.DataFrame):
            if output_preferences.format == "pdf":
                tmp_path = create_pdf(result.return_value)
                return FileResponse(
                    tmp_path,
                    media_type='application/pdf',
                    filename='query_results.pdf'
                )
            # default to CSV for DataFrames
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as tmp:
                result.return_value.to_csv(tmp.name, index=False)
                return FileResponse(
                    tmp.name,
                    media_type='text/csv',
                    filename='query_results.csv'
                )
        
        elif isinstance(result.return_value, (dict, list)):
            if output_preferences.format == "pdf":
                tmp_path = create_pdf(result.return_value)
                return FileResponse(
                    tmp_path,
                    media_type='application/pdf',
                    filename='query_results.pdf'
                )
            # default to JSON
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
                json.dump(result.return_value, tmp)
                return FileResponse(
                    tmp.name,
                    media_type='application/json',
                    filename='query_results.json'
                )
        
        else:  # string or other types
            if output_preferences.format == "pdf":
                tmp_path = create_pdf(result.return_value)
                return FileResponse(
                    tmp_path,
                    media_type='application/pdf',
                    filename='query_results.pdf'
                )
            # default to TXT
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as tmp:
                tmp.write(str(result.return_value))
                return FileResponse(
                    tmp.name,
                    media_type='text/plain',
                    filename='query_results.txt'
                )
    
    elif output_preferences.destination_url:
        # Handle destination URL upload
        await handle_destination_upload(
            result.return_value, 
            output_preferences.destination_url
        )
        return {
            "message": "success",
            "result": "Data uploaded successfully"
        }

    # Default response
    return {
        "message": "success",
        "result": (result.return_value.to_dict(orient='records') 
                if hasattr(result.return_value, 'to_dict') 
                else result.return_value)
    }

@router.post("/process_query", response_model=ProcessedQueryResult)
async def process_query_endpoint(
    request: QueryRequest,
):
    try:
        # Create a session directory for this request
        session_dir = temp_file_manager.get_temp_dir()
        
        # Process files using the helper function
        processed_data = await preprocess_files(
            files=request.files,
            web_urls=request.web_urls,
            query=request.query,
            session_dir=session_dir
        )
        
        # Process the query with the processed data
        sandbox = EnhancedPythonInterpreter()
        result = process_query(
            query=request.query,
            sandbox=sandbox,
            data=processed_data
        )
        
        result.return_value_snapshot = _create_return_value_snapshot(result.return_value)

        
        # Clean up temporary files
        try:
            temp_file_manager.cleanup()
        except Exception as e:
            logging.error(f"Error cleaning up temporary files: {str(e)}")
        
        print("\nOriginal Query:", 
          result.original_query, "\nResult:", 
          "\nOutput:", result.print_output, 
          "\nCode:", result.code, 
          "\nError:", result.error, 
          "\nReturn Value Snapshot:", result.return_value_snapshot, 
          "\nTimed Out:", result.timed_out, 
          "\n\n")
        
        # Use the helper function to handle output preferences
        return await handle_output_preferences(result, request.output_preferences)

    except Exception as e:
        return {"message": "error", "result": str(e)}
