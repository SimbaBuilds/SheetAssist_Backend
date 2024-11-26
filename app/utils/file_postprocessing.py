from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import tempfile
import pandas as pd
from typing import Any
import json
from docx import Document
import openpyxl
from app.schemas import FileDataInfo
from typing import List
from app.utils.llm import file_namer
import csv
from app.utils.online_document_integrations import DocumentIntegrations
import logging
from fastapi import HTTPException

def prepare_dataframe(data: Any) -> pd.DataFrame:
    """Prepare and standardize any data type into a clean DataFrame"""
    
    # Convert input to DataFrame if it isn't already
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    elif isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame({'Value': data})
    else:
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
            # Convert lists or dicts in cells to strings
            if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
                df[col] = df[col].apply(lambda x: str(x) if isinstance(x, (list, dict)) else x)
            
            # Ensure numeric columns are properly formatted
            if df[col].dtype in ['float64', 'int64']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any problematic characters from string columns
        str_columns = df.select_dtypes(include=['object']).columns
        for col in str_columns:
            df[col] = df[col].astype(str).str.replace('\x00', '')
            
    except Exception as e:
        print(f"Warning during DataFrame preparation: {str(e)}")
        
    return df

def create_csv(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create CSV file from prepared DataFrame"""
    filename = file_namer(query, old_data)
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

def create_xlsx(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create Excel file from various data types"""
    filename = file_namer(query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.xlsx')
    
    if isinstance(new_data, pd.DataFrame):
        new_data.to_excel(tmp_path, index=False)
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        
        if isinstance(new_data, (dict, list)):
            # Convert to string and write as single cell
            ws['A1'] = json.dumps(new_data, indent=2)
        else:
            ws['A1'] = str(new_data)
            
        wb.save(tmp_path)
    
    return tmp_path

def create_pdf(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create PDF file from various data types"""
    filename = file_namer(query, old_data)
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

def create_docx(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create Word document from various data types"""
    filename = file_namer(query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.docx')
    doc = Document()
    
    if isinstance(new_data, pd.DataFrame):
        # Add table
        table = doc.add_table(rows=len(new_data)+1, cols=len(new_data.columns))
        
        # Add headers
        for j, column in enumerate(new_data.columns):
            table.cell(0, j).text = str(column)
            
        # Add data
        for i, row in enumerate(new_data.values):
            for j, cell in enumerate(row):
                table.cell(i+1, j).text = str(cell)
                
    elif isinstance(new_data, (dict, list)):
        doc.add_paragraph(json.dumps(new_data, indent=2))
    else:
        doc.add_paragraph(str(new_data))
    
    doc.save(tmp_path)
    return tmp_path

def create_txt(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create text file from various data types"""
    filename = file_namer(query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.txt')
    
    with open(tmp_path, 'w', encoding='utf-8') as f:
        if isinstance(new_data, pd.DataFrame):
            f.write(new_data.to_string())
        elif isinstance(new_data, (dict, list)):
            f.write(json.dumps(new_data, indent=2))
        else:
            f.write(str(new_data))
    
    return tmp_path

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
