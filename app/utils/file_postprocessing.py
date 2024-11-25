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

def create_csv(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create CSV file from various data types"""
    filename = file_namer(query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.csv')
    
    if isinstance(new_data, pd.DataFrame):
        # Ensure clean CSV output with proper column separation
        new_data.reset_index(drop=True).to_csv(
            tmp_path,
            index=False,  # Don't include row numbers
            sep=',',
            encoding='utf-8',
            quoting=1,  # Quote all non-numeric values
            quotechar='"',  # Use double quotes for text fields
            line_terminator='\n'  # Ensure proper line endings
        )
    else:
        # Convert other data types to DataFrame first, then to CSV
        print("new_data is not a DataFrame")
        if isinstance(new_data, (dict, list)):
            if isinstance(new_data, dict):
                df = pd.DataFrame([new_data])
            else:
                if all(isinstance(item, dict) for item in new_data):
                    df = pd.DataFrame(new_data)
                else:
                    df = pd.DataFrame({'Value': new_data})
            df.to_csv(tmp_path, index=False, sep=',', encoding='utf-8')
        else:
            # For simple types, create a single-column CSV
            pd.DataFrame([new_data]).to_csv(tmp_path, index=False, header=False, sep=',', encoding='utf-8')
    
    return tmp_path
