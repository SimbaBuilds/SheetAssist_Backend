from fastapi import APIRouter, UploadFile, File, Form
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
        
        # Handle both tuple returns and single items
        if isinstance(result.return_value, tuple):
            # If it's a tuple, convert each DataFrame to dict
            return {
                "message": "success",
                "result": [df.to_dict(orient='records') if hasattr(df, 'to_dict') else df 
                        for df in result.return_value]
            }
        else:
            # Handle single return value
            return {
                "message": "success",
                "result": (result.return_value.to_dict(orient='records') 
                        if hasattr(result.return_value, 'to_dict') 
                        else result.return_value)
            }
       
    except Exception as e:
        return {"message": "error", "result": str(e)}
