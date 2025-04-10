from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import pandas as pd
from typing import Any, Optional, List, Tuple
import json
from docx import Document
from app.schemas import FileDataInfo, SandboxResult, QueryRequest, BatchProcessingFileInfo
from typing import List, Tuple
from app.utils.llm_service import LLMService
from app.utils.data_processing import get_data_snapshot
import csv
import logging
from fastapi import HTTPException
import pandas as pd
import json
import logging
from typing import Any
from app.utils.auth import SupabaseClient
from app.utils.google_integration import GoogleIntegration
from app.utils.microsoft_integration import MicrosoftIntegration
import pickle
import os
from datetime import datetime, UTC




def prepare_dataframe(data: Any) -> pd.DataFrame:
    """Prepare and standardize any data type into a clean DataFrame"""
    
    # Step 1: Handle string input that contains DataFrame representation
    if isinstance(data, str) or (isinstance(data, list) and len(data) == 1 and isinstance(data[0], str)):
        print("\nData is a string or a list of strings\n")
        input_str = data[0] if isinstance(data, list) else data
        
        # Step 2: Clean up the DataFrame string representation
        # Remove shape information and parentheses
        cleaned_str = input_str.replace('[1 rows x 10 columns],)', '').strip('()')
        
        # Step 3: Split the cleaned string into rows
        rows = [row.strip() for row in cleaned_str.split('\n') if row.strip()]
        
        # Step 4: Split each row into columns
        # First row contains headers
        headers = rows[0].split()
        
        # Process data rows
        data_rows = []
        for row in rows[1:]:
            # Split on multiple spaces to separate columns
            values = [val.strip() for val in row.split('  ') if val.strip()]
            data_rows.append(values)
        
        # Step 5: Create DataFrame from processed data
        df = pd.DataFrame(data_rows, columns=headers)
        
    elif isinstance(data, pd.DataFrame):
        print("\nData is already a DataFrame\n")
        df = data.copy()
    elif isinstance(data, dict):
        print("\nData is a dictionary\n")
        df = pd.DataFrame([data])
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        print("\nData is a list of dictionaries\n")
        df = pd.DataFrame(data)
    elif isinstance(data, tuple):
        # If first element is a DataFrame, return that
        print(f"\nData is of type {type(data).__name__}\n")
        if isinstance(data[0], pd.DataFrame):
            df = data[0]
        # Otherwise convert tuple to DataFrame
        else:
            df = pd.DataFrame([data], columns=[f'Value_{i}' for i in range(len(data))])
    else:
        print("\nData is a single value\n")
        df = pd.DataFrame({'Value': [data]})
    
    # Clean and standardize the DataFrame
    try:
        # Clean column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r'[^\w\s-]', '_', regex=True)
        
        # Handle missing values
        df = df.fillna('')
        
        # Convert problematic data types
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            
            # Ensure numeric columns are properly formatted
            try:
                if df[col].dtype in ['float64', 'int64'] or df[col].str.match(r'^-?\d*\.?\d+$').all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        # Remove any problematic characters from string columns
        str_columns = df.select_dtypes(include=['object']).columns
        for col in str_columns:
            df[col] = df[col].astype(str).str.replace('\x00', '')
            
    except Exception as e:
        print(f"Warning during DataFrame preparation: {str(e)}")
        
    return df

def prepare_text(data: Any) -> str:
    """Prepare and standardize any data type into a text string"""
    
    if isinstance(data, tuple):
        print(f"\nData is of type {type(data).__name__}\n")
        if isinstance(data[0], str):
            extracted_text = data[0]
        elif isinstance(data[0], pd.DataFrame):
            extracted_text = data[0].to_string()
        else:
            # Convert tuple to string without trying to use columns parameter
            extracted_text = str(list(data))
    elif isinstance(data, str):
        extracted_text = data
    else:
        extracted_text = str(data)
        
    return extracted_text

async def create_csv(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create CSV file from prepared DataFrame"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{filename}_{timestamp}.csv")
    
    # Prepare the DataFrame
    df = prepare_dataframe(new_data)
    
    # Write to CSV with consistent parameters
    try:
        df.to_csv(
            tmp_path,
            index=False,
            encoding='utf-8',
            sep=',',
            quoting=csv.QUOTE_MINIMAL,
            quotechar='"',
            escapechar='\\',
            lineterminator='\n',
            float_format='%.2f'
        )
    except Exception as e:
        print(f"Error writing CSV: {str(e)}")
        raise
    
    return tmp_path

async def create_xlsx(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create Excel file from various data types"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{filename}_{timestamp}.xlsx")
    df = prepare_dataframe(new_data)
    df.to_excel(tmp_path, index=False)

    return tmp_path

async def create_pdf(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create PDF file from various data types"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{filename}_{timestamp}.pdf")
    doc = SimpleDocTemplate(tmp_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(Paragraph("Query Results", styles['Heading1']))
    elements.append(Spacer(1, 12))

    if isinstance(new_data, pd.DataFrame):
        # Convert DataFrame to table
        table_data = [new_data.columns.tolist()] + new_data.values.tolist()
        t = Table(table_data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        elements.append(t)
    
    elif isinstance(new_data, (dict, list)):
        # Convert dict/list to formatted text
        json_str = json.dumps(new_data, indent=2)
        elements.append(Paragraph(json_str, styles['Code']))
    
    elif isinstance(new_data, str):
        # Handle plain text
        elements.append(Paragraph(new_data, styles['Normal']))
    
    doc.build(elements)
    return tmp_path

async def create_docx(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create Word document from various data types"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{filename}_{timestamp}.docx")
    doc = Document()
    
    extracted_text = prepare_text(new_data)
    doc.add_paragraph(str(extracted_text))
    doc.save(tmp_path)
    return tmp_path

async def create_txt(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create text file from various data types"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_path = os.path.join(tempfile.gettempdir(), f"{filename}_{timestamp}.txt")
    extracted_text = prepare_text(new_data)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    return tmp_path




async def handle_destination_upload(data: Any, request: QueryRequest, old_data: List[FileDataInfo], supabase: SupabaseClient, user_id: str) -> bool:
    """Upload data to various destination types"""
    try:
        # Process tuple data if present
        if isinstance(data, tuple):
            if isinstance(data[0], pd.DataFrame):
                data = data[0]
            elif isinstance(data[0], str):
                data = data[0]
            else:
                data = pd.DataFrame([data], columns=[f'Value_{i}' for i in range(len(data))])

        llm_service = LLMService()
        url_lower = request.output_preferences.destination_url.lower()
        
        if "docs.google.com" in url_lower:
            g_integration = GoogleIntegration(
                supabase=supabase,
                user_id=user_id,
                picker_token=request.output_preferences.picker_token if "docs.google.com" in request.output_preferences.destination_url.lower() else None
            ) 
            if request.output_preferences.modify_existing:
                return await g_integration.append_to_current_google_sheet(
                    data, 
                    request.output_preferences.destination_url,
                    request.output_preferences.sheet_name
                )
            else:
                provider, suggested_name = await llm_service.execute_with_fallback(
                    "file_namer",
                    request.query,
                    old_data
                )
                return await g_integration.append_to_new_google_sheet(
                    data, 
                    request.output_preferences.destination_url, 
                    suggested_name
                )
        
        elif "onedrive" in url_lower or "sharepoint.com" in url_lower:
            msft_integration = MicrosoftIntegration(
                supabase=supabase,
                user_id=user_id,
                # picker_token=request.output_preferences.picker_token if any(x in request.output_preferences.destination_url.lower() for x in ["onedrive", "sharepoint.com"]) else None
            )
            if request.output_preferences.modify_existing:
                return await msft_integration.append_to_current_office_sheet(
                    data, 
                    request.output_preferences.destination_url, 
                    request.output_preferences.sheet_name
                )
            else:
                provider, suggested_name = await llm_service.execute_with_fallback(
                    "file_namer",
                    request.query,
                    old_data
                )
                return await msft_integration.append_to_new_office_sheet(
                    data, 
                    request.output_preferences.destination_url, 
                    suggested_name
                )
    
        raise ValueError(f"Unsupported destination sheet type: {request.output_preferences.destination_url}")
    
    except Exception as e:
        logging.error(f"Failed to upload to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def handle_download(result: SandboxResult, request: QueryRequest, preprocessed_data: List[FileDataInfo]) -> Tuple[str, str]:
    # Get the desired output format, defaulting based on data type
    output_format = request.output_preferences.format


    llm_service = LLMService()
    # Create temporary file in requested format
    if output_format == 'pdf':
        tmp_path = await create_pdf(result.return_value, request.query, preprocessed_data, llm_service)
        media_type = 'application/pdf'
    elif output_format == 'xlsx':
        tmp_path = await create_xlsx(result.return_value, request.query, preprocessed_data, llm_service)
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif output_format == 'docx':
        tmp_path = await create_docx(result.return_value, request.query, preprocessed_data, llm_service)
        media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif output_format == 'txt':
        tmp_path = await create_txt(result.return_value, request.query, preprocessed_data, llm_service)
        media_type = 'text/plain'
    else:  # csv
        tmp_path = await create_csv(result.return_value, request.query, preprocessed_data, llm_service)
        media_type = 'text/csv'

    return tmp_path, media_type

async def handle_batch_destination_upload(
    data: Any,
    request: QueryRequest,
    old_data: List[FileDataInfo],
    supabase: SupabaseClient,
    user_id: str,
    llm_service: LLMService,
    is_first_chunk: bool,
    job_id: str
) -> bool:
    """Handle destination upload for batch processing"""
    try:
        # Process data if it's a tuple
        processed_data = data
        if isinstance(data, tuple):
            if isinstance(data[0], pd.DataFrame):
                processed_data = data[0]
            elif isinstance(data[0], str):
                processed_data = data[0]
            else:
                processed_data = pd.DataFrame([data], columns=[f'Value_{i}' for i in range(len(data))])
        
        url_lower = request.output_preferences.destination_url.lower()
        suggested_name = None

        # Initialize integrations
        g_integration = None
        msft_integration = None
        if "docs.google.com" in url_lower:
            g_integration = GoogleIntegration(
                supabase=supabase,
                user_id=user_id,
                picker_token=request.output_preferences.picker_token
            )
        elif "onedrive" in url_lower or "sharepoint.com" in url_lower:
            msft_integration = MicrosoftIntegration(
                supabase=supabase,
                user_id=user_id
            )

        # For first chunk with new sheet creation
        if is_first_chunk and not request.output_preferences.modify_existing:
            print(f"Creating new sheet first chunk: {is_first_chunk} and preferences: {request.output_preferences.modify_existing}")
            provider, suggested_name = await llm_service.execute_with_fallback(
                "file_namer",
                request.query,
                old_data
            )
            
            # Get current job to preserve existing output_preferences
            current_job = supabase.table("jobs").select("*").eq("job_id", job_id).execute()
            if not current_job.data:
                raise HTTPException(status_code=404, detail="Job not found")
            
            # Merge the new sheet_name with existing preferences
            current_preferences = current_job.data[0]["output_preferences"]
            current_preferences["sheet_name"] = suggested_name
            
            # Update with merged preferences
            supabase.table("jobs").update({
                "output_preferences": current_preferences
            }).eq("job_id", job_id).execute()

            if g_integration:
                return await g_integration.append_to_new_google_sheet(
                    processed_data,
                    request.output_preferences.destination_url,
                    suggested_name
                )
            elif msft_integration:
                return await msft_integration.append_to_new_office_sheet(
                    processed_data,
                    request.output_preferences.destination_url,
                    suggested_name
                )
        
        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = job_response.data[0]
    
        if request.output_preferences.modify_existing:
            sheet_name = request.output_preferences.sheet_name
        else:
            sheet_name = job["output_preferences"]["sheet_name"]
      
        # For subsequent chunks or when modifying existing
        if g_integration:
            print(f"Appending to existing sheet.  First chunk: {is_first_chunk} Sheet Name: {sheet_name}")
            return await g_integration.append_to_current_google_sheet(
                processed_data,
                request.output_preferences.destination_url,
                sheet_name
            )
        elif msft_integration:
            print(f"Appending to existing sheet.  First chunk: {is_first_chunk} Sheet Name: {sheet_name}")
            return await msft_integration.append_to_current_office_sheet(
                processed_data,
                request.output_preferences.destination_url,
                sheet_name
            )
            
        raise ValueError(f"Unsupported destination sheet type: {request.output_preferences.destination_url}")
        
    except Exception as e:
        logging.error(f"Failed to upload batch to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch upload failed: {str(e)}")

async def handle_batch_chunk_result(
    result: SandboxResult,
    request_data: QueryRequest,
    preprocessed_data: List[FileDataInfo],
    supabase: SupabaseClient,
    user_id: str,
    job_id: str,
    session_dir: str,
    current_chunk: int,
    total_chunks: int,
    num_images_processed: int,
) -> Tuple[str, Optional[str], Optional[str]]:
    """Handle the result of a batch chunk processing."""
    try:
        llm_service = LLMService()
        
        # Extract data from tuple if needed
        processed_data = result.return_value
        if isinstance(processed_data, tuple):
            if isinstance(processed_data[0], pd.DataFrame):
                processed_data = processed_data[0]
            elif isinstance(processed_data[0], str):
                processed_data = processed_data[0]
            else:
                processed_data = pd.DataFrame([processed_data], columns=[f'Value_{i}' for i in range(len(processed_data))])

        # Initialize return values
        status = "processing"
        result_file_path = None
        result_media_type = None

        # Get data snapshot for database update
        result_snapshot = get_data_snapshot(
            processed_data,
            type(processed_data).__name__
        )

        job_response = supabase.table("jobs").select("*").eq("job_id", job_id).eq("user_id", user_id).execute()
        
        if not job_response.data:
            raise HTTPException(status_code=404, detail="Job not found")
            
        job = job_response.data[0]
        
        
        # Prepare base update data
        update_data = {
            "result_snapshot": result_snapshot,
            "query": request_data.query
        }

        # Set started_at for first chunk
        if current_chunk == 0:
            update_data["started_at"] = datetime.now(UTC).isoformat()

        # Handle based on output preference type
        if request_data.output_preferences.type == "download":
            # Process final chunk
            if current_chunk == total_chunks - 1:
                try:
                    tmp_path, media_type = await handle_download(
                        SandboxResult(
                            return_value=processed_data,
                            error=None,
                            original_query=request_data.query,
                            print_output="",
                            code="",
                            timed_out=False
                        ),
                        request_data,
                        preprocessed_data,
                    )
                    result_file_path = str(tmp_path)
                    result_media_type = media_type
                    
                    # Update final data
                    update_data.update({
                        "completed_at": datetime.now(UTC).isoformat(),
                        "result_file_path": result_file_path,
                        "result_media_type": result_media_type
                    })
                    supabase.table("jobs").update(update_data).eq("job_id", job_id).execute()

                except Exception as e:
                    error_msg = f"Error creating download for batch result: {str(e)}"
                    logging.error(error_msg)
                    update_data.update({
                        "status": "error",
                        "error_message": error_msg,
                        "completed_at": datetime.now(UTC).isoformat()
                    })
                    supabase.table("jobs").update(update_data).eq("job_id", job_id).execute()
                    raise ValueError(error_msg)

        elif request_data.output_preferences.type == "online":
            try:
                # Handle online output
                is_first_chunk = current_chunk == 0
                await handle_batch_destination_upload(
                    processed_data,
                    request_data,
                    preprocessed_data,
                    supabase,
                    user_id,
                    llm_service,
                    is_first_chunk,
                    job_id
                )

                # Mark as completed if last chunk
                if current_chunk == total_chunks - 1:
                    update_data.update({
                        "completed_at": datetime.now(UTC).isoformat()
                    })
            except Exception as e:
                error_msg = f"Error uploading batch result to destination: {str(e)}"
                logging.error(error_msg)
                update_data.update({
                    "status": "error",
                    "error_message": error_msg,
                    "completed_at": datetime.now(UTC).isoformat()
                })
                supabase.table("jobs").update(update_data).eq("job_id", job_id).execute()
                raise ValueError(error_msg)

        # Update job status in database
        supabase.table("jobs").update(update_data).eq("job_id", job_id).execute()

        return status, result_file_path, result_media_type

    except Exception as e:
        error_msg = f"Error in batch chunk postprocessing: {str(e)}"
        logging.error(error_msg)
        error_update = {
            "status": "error",
            "error_message": error_msg,
            "completed_at": datetime.now(UTC).isoformat()
        }
        supabase.table("jobs").update(error_update).eq("job_id", job_id).execute()
        raise ValueError(error_msg)