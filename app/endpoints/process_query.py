from fastapi import APIRouter, UploadFile, File, Form
from typing import List, Optional
from pydantic import BaseModel
from app.utils.file_preprocessing import FilePreprocessor
from app.utils.process_query import process_query
from app.utils.sandbox import EnhancedPythonInterpreter
from app.class_schemas import (
    TabularDataInfo, JsonDataInfo, TextDataInfo, 
    ImageDataInfo, WebDataInfo
)
import pandas as pd
import json

router = APIRouter()

class QueryRequest(BaseModel):
    web_urls: Optional[List[str]] = []
    files: Optional[List[UploadFile]] = []
    query: str

@router.post("/process_query")
async def process_query_endpoint(
    request: QueryRequest,
):
    try:
        # Initialize preprocessor and data storage
        preprocessor = FilePreprocessor()
        processed_data = []
        
        # Process web URLs if provided
        for url in request.web_urls:
            try:
                df = preprocessor.process_web_url(url)
                processed_data.append(
                    WebDataInfo(
                        url=url,
                        df=df,
                        source_type='google_sheets' if 'docs.google.com' in url else 'excel_web',
                        converted_csv_path=None  # You might want to save to CSV and store path
                    )
                )
            except Exception as e:
                return {"status": "error", "message": f"Error processing URL {url}: {str(e)}"}
            
        # Process uploaded files
        if request.files:
            for file in request.files:
                file_ext = file.filename.split('.')[-1].lower()
                
                # Handle tabular data (Excel and CSV)
                if file_ext in ['xlsx', 'xls']:
                    df = preprocessor.process_excel(file.file)
                    processed_data.append(
                        TabularDataInfo(
                            df=df,
                            snapshot=df.head(10),
                            file_name=file.filename,
                            data_type="DataFrame"
                        )
                    )
                elif file_ext == 'csv':
                    df = preprocessor.process_csv(file.file)
                    processed_data.append(
                        TabularDataInfo(
                            df=df,
                            snapshot=df.head(10),
                            file_name=file.filename,
                            data_type="DataFrame"
                        )
                    )
                
                # Handle JSON data
                elif file_ext == 'json':
                    json_str = preprocessor.process_json(file.file)
                    processed_data.append(
                        JsonDataInfo(
                            content=json.loads(json_str),
                            string_representation=json_str,
                            file_name=file.filename
                        )
                    )
                
                # Handle text files
                elif file_ext == 'txt':
                    content = preprocessor.process_text(file.file)
                    processed_data.append(
                        TextDataInfo(
                            content=content,
                            file_name=file.filename,
                            data_type="text",
                            original_format="txt"
                        )
                    )
                
                # Handle Word documents
                elif file_ext == 'docx':
                    content = preprocessor.process_docx(file.file)
                    processed_data.append(
                        TextDataInfo(
                            content=content,
                            file_name=file.filename,
                            data_type="text",
                            original_format="docx"
                        )
                    )
                
                # Handle images
                elif file_ext in ['png', 'jpeg', 'jpg']:
                    if file_ext == 'png':
                        # Convert PNG to JPEG
                        jpeg_path = preprocessor.process_image(file.file)
                        processed_data.append(
                            ImageDataInfo(
                                image_data=file.file,
                                file_name=file.filename,
                                original_format="png",
                                converted_jpeg_path=jpeg_path
                            )
                        )
                    else:
                        processed_data.append(
                            ImageDataInfo(
                                image_data=file.file,
                                file_name=file.filename,
                                original_format="jpeg"
                            )
                        )
                
                # Handle PDF files (assuming PDF processing is implemented in FilePreprocessor)
                elif file_ext == 'pdf':
                    # TODO: Implement PDF processing
                    pass
                
        
        # Initialize sandbox environment
        sandbox = EnhancedPythonInterpreter()
        
        # Process query using the processed data
        result = process_query(
            query=request.query,
            sandbox=sandbox,
            data=processed_data
        )
        
        if result.error:
            return {"status": "error", "message": result.error}
            
        if result.return_value is not None:
            df_snapshot = result.return_value.head(10)
        else:
            df_snapshot = "None"
        
        print("\nOriginal Query:", 
          result.original_query, "\nResult:", 
          "\nOutput:", result.print_output, 
          "\nCode:", result.code, 
          "\nError:", result.error, 
          "\nReturn Value Snapshot:", df_snapshot, 
              "\nTimed Out:", result.timed_out, 
              "\n\n")
        
        return {
            "status": "success",
            "data": result.return_value.to_dict(orient='records')
        }
       
    except Exception as e:
        return {"status": "error", "message": str(e)}
