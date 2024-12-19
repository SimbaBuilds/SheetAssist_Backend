from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import pandas as pd
from typing import Any
import json
from docx import Document
from app.schemas import FileDataInfo, SandboxResult, QueryRequest
from typing import List, Tuple
from app.utils.llm_service import LLMService
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
    
    if isinstance(data, tuple):
        print(f"\nData is of type {type(data).__name__}\n")
        extracted_text = ""
        if isinstance(data[0], str):
            extracted_text = data[0]
        else:
            extracted_text = str([data], columns=[f'Value_{i}' for i in range(len(data))])

    elif isinstance(data, str):
        extracted_text = data
    else:
        extracted_text = str(data)
        
    return extracted_text

async def create_csv(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create CSV file from prepared DataFrame"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.csv')
    
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
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.xlsx')
    df = prepare_dataframe(new_data)
    df.to_excel(tmp_path, index=False)

    return tmp_path

async def create_pdf(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create PDF file from various data types"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.pdf')
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
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.docx')
    doc = Document()
    
    extracted_text = prepare_text(new_data)
    doc.add_paragraph(str(extracted_text))
    doc.save(tmp_path)
    return tmp_path

async def create_txt(new_data: Any, query: str, old_data: List[FileDataInfo], llm_service: LLMService) -> str:
    """Create text file from various data types"""
    provider, filename = await llm_service.execute_with_fallback("file_namer", query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.txt')
    extracted_text = prepare_text(new_data)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    return tmp_path




async def handle_destination_upload(data: Any, request: QueryRequest, old_data: List[FileDataInfo], supabase: SupabaseClient, user_id: str, llm_service: LLMService) -> bool:
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
        

        g_integration = GoogleIntegration(supabase, user_id)
        msft_integration = MicrosoftIntegration(supabase, user_id)

        url_lower = request.output_preferences.destination_sheet.lower()
        
        if "docs.google.com" in url_lower:
            if request.output_preferences.modify_existing:
                return await g_integration.append_to_current_google_sheet(
                    data, 
                    request.output_preferences.destination_sheet,
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
                    request.output_preferences.destination_sheet, 
                    suggested_name
                )
        
        elif "onedrive" in url_lower or "sharepoint.com" in url_lower:
            if request.output_preferences.modify_existing:
                return await msft_integration.append_to_current_office_sheet(
                    data, 
                    request.output_preferences.destination_sheet, 
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
                    request.output_preferences.destination_sheet, 
                    suggested_name
                )
    
        raise ValueError(f"Unsupported destination sheet type: {request.output_preferences.destination_sheet}")
    
    except Exception as e:
        logging.error(f"Failed to upload to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

async def handle_download(result: SandboxResult, request: QueryRequest, preprocessed_data: List[FileDataInfo], llm_service: LLMService) -> Tuple[str, str]:
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