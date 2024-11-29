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
from app.schemas import FileDataInfo, SandboxResult, QueryRequest
from typing import List, Tuple
from app.utils.llm import file_namer
import csv
import logging
from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
import pandas as pd
import json
import logging
from typing import Any
import os
from app.utils.auth import SupabaseClient

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
    df = prepare_dataframe(new_data)
    df.to_excel(tmp_path, index=False)

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
    
    extracted_text = prepare_text(new_data)
    doc.add_paragraph(str(extracted_text))
    doc.save(tmp_path)
    return tmp_path

def create_txt(new_data: Any, query: str, old_data: List[FileDataInfo]) -> str:
    """Create text file from various data types"""
    filename = file_namer(query, old_data)
    tmp_path = tempfile.mktemp(prefix=f"{filename}_", suffix='.txt')
    extracted_text = prepare_text(new_data)
    with open(tmp_path, 'w', encoding='utf-8') as f:
        f.write(extracted_text)
    return tmp_path

class DocumentIntegrations:
    def __init__(self, google_refresh_token: str):
        # Google credentials
        self.google_creds = Credentials(
            token=None,
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            refresh_token=google_refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            scopes=['https://www.googleapis.com/auth/documents',
                   'https://www.googleapis.com/auth/spreadsheets']
        )
        
        # Microsoft credentials
        self.ms_credential = ClientSecretCredential(
            tenant_id=os.getenv('MS_TENANT_ID'),
            client_id=os.getenv('MS_CLIENT_ID'),
            client_secret=os.getenv('MS_CLIENT_SECRET')
        )
        self.ms_scopes = ['https://graph.microsoft.com/.default']
        self.ms_client = GraphServiceClient(
            credentials=self.ms_credential,
            scopes=self.ms_scopes
        )

    async def append_to_google_doc(self, data: Any, doc_url: str) -> bool:
        """Append data to Google Doc"""
        try:
            # Extract document ID from URL
            doc_id = doc_url.split('/d/')[1].split('/')[0]
            
            # Create Google Docs service
            service = build('docs', 'v1', credentials=self.google_creds)
            
            # Format content based on data type
            if isinstance(data, pd.DataFrame):
                content = data.to_string()
            elif isinstance(data, (dict, list)):
                content = json.dumps(data, indent=2)
            else:
                content = str(data)
            
            # Append content to end of document
            requests = [{
                'insertText': {
                    'location': {'index': 1},  # End of document
                    'text': f"\n\n{content}"
                }
            }]
            
            service.documents().batchUpdate(
                documentId=doc_id,
                body={'requests': requests}
            ).execute()
            
            return True
            
        except Exception as e:
            logging.error(f"Google Docs append error: {str(e)}")
            raise

    async def append_to_google_sheet(self, data: Any, sheet_url: str) -> bool:
        """Append data to Google Sheet"""
        try:
            # Extract spreadsheet ID from URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # Create Google Sheets service with proper scopes
            service = build('sheets', 'v4', credentials=self.google_creds)
            
            # Get sheet name from URL or fetch first sheet if not specified
            sheet_name = None
            if '#gid=' in sheet_url:
                gid = sheet_url.split('#gid=')[1]
                # Get spreadsheet metadata to find sheet name
                sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
                for sheet in sheet_metadata.get('sheets', ''):
                    if sheet.get('properties', {}).get('sheetId') == int(gid):
                        sheet_name = sheet['properties']['title']
                        break
            
            if not sheet_name:
                # If no specific sheet found, get the first sheet name
                sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
                sheet_name = sheet_metadata['sheets'][0]['properties']['title']
            
            # Format data for sheets
            if isinstance(data, pd.DataFrame):
                values = [data.columns.tolist()] + data.values.tolist()
            elif isinstance(data, (dict, list)):
                if isinstance(data, dict):
                    values = [[k, str(v)] for k, v in data.items()]
                else:
                    values = [[str(item)] for item in data]
            else:
                values = [[str(data)]]
            
            # Append data to sheet using values.append
            body = {
                'values': values
            }
            service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range=f"{sheet_name}",  # Use the detected sheet name
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            return True
            
        except Exception as e:
            logging.error(f"Google Sheets append error: {str(e)}")
            raise

    async def add_to_new_google_sheet(self, data: Any, sheet_name: str) -> bool:
        """Add data to a new Google Sheet"""
        pass

    async def append_to_office_doc(self, data: Any, doc_url: str) -> bool:
        """Append data to Office Word Online document"""
        try:
            # Initialize Microsoft Graph client
            graph_client = GraphServiceClient(self.ms_credential, self.ms_scopes)
            
            # Extract document ID from URL
            doc_id = doc_url.split('/')[-2]
            
            # Format content based on data type
            if isinstance(data, pd.DataFrame):
                content = data.to_string()
            elif isinstance(data, (dict, list)):
                content = json.dumps(data, indent=2)
            else:
                content = str(data)
            
            # Append content to document
            graph_client.documents[doc_id].content.append(content)
            
            return True
            
        except Exception as e:
            logging.error(f"Office Word append error: {str(e)}")
            raise

    async def append_to_office_sheet(self, data: Any, sheet_url: str) -> bool:
        """Append data to Office Excel Online workbook"""
        try:
            # Initialize Microsoft Graph client
            graph_client = GraphServiceClient(self.ms_credential, self.ms_scopes)
            
            # Extract workbook ID from URL
            workbook_id = sheet_url.split('/')[-2]
            
            # Create new worksheet
            worksheet = graph_client.workbook.worksheets.add("Query Results")
            
            # Format data for Excel
            if isinstance(data, pd.DataFrame):
                values = [data.columns.tolist()] + data.values.tolist()
            elif isinstance(data, (dict, list)):
                if isinstance(data, dict):
                    values = [[k, str(v)] for k, v in data.items()]
                else:
                    values = [[str(item)] for item in data]
            else:
                values = [[str(data)]]
            
            # Write data to worksheet
            worksheet.range('A1').values = values
            
            return True
            
        except Exception as e:
            logging.error(f"Office Excel append error: {str(e)}")
            raise 

    async def add_to_new_office_sheet(self, data: Any, sheet_name: str) -> bool:
        """Add data to a new Office Excel Online workbook"""
        pass

async def handle_destination_upload(data: Any, destination_url: str, supabase: SupabaseClient, user_id: str) -> bool:
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
        
        response = supabase.table('user_documents_access') \
            .select('refresh_token') \
            .match({'user_id': user_id, 'provider': 'google'}) \
            .execute()
        
        if not response.data or len(response.data) == 0:
            print(f"No Google token found for user {user_id}")
            return None
        google_refresh_token = response.data[0]['refresh_token']
        doc_integrations = DocumentIntegrations(google_refresh_token)
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

def handle_download(result: SandboxResult, request: QueryRequest, preprocessed_data: List[FileDataInfo]) -> Tuple[str, str]:
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
        tmp_path = create_pdf(result.return_value, request.query, preprocessed_data)
        media_type = 'application/pdf'
    elif output_format == 'xlsx':
        tmp_path = create_xlsx(result.return_value, request.query, preprocessed_data)
        media_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif output_format == 'docx':
        tmp_path = create_docx(result.return_value, request.query, preprocessed_data)
        media_type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    elif output_format == 'txt':
        tmp_path = create_txt(result.return_value, request.query, preprocessed_data)
        media_type = 'text/plain'
    else:  # csv
        tmp_path = create_csv(result.return_value, request.query, preprocessed_data)
        media_type = 'text/csv'

    return tmp_path, media_type