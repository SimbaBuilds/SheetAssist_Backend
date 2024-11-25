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

def create_pdf(data: Any) -> str:
    """Create PDF file from various data types"""
    tmp_path = tempfile.mktemp(suffix='.pdf')
    doc = SimpleDocTemplate(tmp_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Add title
    elements.append(Paragraph("Query Results", styles['Heading1']))
    elements.append(Spacer(1, 12))

    if isinstance(data, pd.DataFrame):
        # Convert DataFrame to table
        table_data = [data.columns.tolist()] + data.values.tolist()
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
    
    elif isinstance(data, (dict, list)):
        # Convert dict/list to formatted text
        json_str = json.dumps(data, indent=2)
        elements.append(Paragraph(json_str, styles['Code']))
    
    elif isinstance(data, str):
        # Handle plain text
        elements.append(Paragraph(data, styles['Normal']))
    
    doc.build(elements)
    return tmp_path

def create_xlsx(data: Any) -> str:
    """Create Excel file from various data types"""
    tmp_path = tempfile.mktemp(suffix='.xlsx')
    
    if isinstance(data, pd.DataFrame):
        data.to_excel(tmp_path, index=False)
    else:
        wb = openpyxl.Workbook()
        ws = wb.active
        
        if isinstance(data, (dict, list)):
            # Convert to string and write as single cell
            ws['A1'] = json.dumps(data, indent=2)
        else:
            ws['A1'] = str(data)
            
        wb.save(tmp_path)
    
    return tmp_path

def create_docx(data: Any) -> str:
    """Create Word document from various data types"""
    tmp_path = tempfile.mktemp(suffix='.docx')
    doc = Document()
    
    if isinstance(data, pd.DataFrame):
        # Add table
        table = doc.add_table(rows=len(data)+1, cols=len(data.columns))
        
        # Add headers
        for j, column in enumerate(data.columns):
            table.cell(0, j).text = str(column)
            
        # Add data
        for i, row in enumerate(data.values):
            for j, cell in enumerate(row):
                table.cell(i+1, j).text = str(cell)
                
    elif isinstance(data, (dict, list)):
        doc.add_paragraph(json.dumps(data, indent=2))
    else:
        doc.add_paragraph(str(data))
    
    doc.save(tmp_path)
    return tmp_path

def create_txt(data: Any) -> str:
    """Create text file from various data types"""
    tmp_path = tempfile.mktemp(suffix='.txt')
    
    with open(tmp_path, 'w', encoding='utf-8') as f:
        if isinstance(data, pd.DataFrame):
            f.write(data.to_string())
        elif isinstance(data, (dict, list)):
            f.write(json.dumps(data, indent=2))
        else:
            f.write(str(data))
    
    return tmp_path

def create_csv(data: Any) -> str:
    """Create CSV file from various data types"""
    tmp_path = tempfile.mktemp(suffix='.csv')
    
    if isinstance(data, pd.DataFrame):
        # Write DataFrame directly to CSV with proper column separation
        data.to_csv(tmp_path, index=False)
    else:
        # Convert other data types to DataFrame first, then to CSV
        if isinstance(data, (dict, list)):
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            else:
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)  # Let pandas handle the conversion
            df.to_csv(tmp_path, index=False)
        else:
            # For simple types, create a single-column CSV without the "Value" header
            pd.DataFrame([str(data)]).to_csv(tmp_path, index=False, header=False)
    
    return tmp_path
