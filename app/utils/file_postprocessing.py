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
from app.schemas import FileDataInfo, SandboxResult, QueryRequest, OutputPreferences
from typing import List, Tuple
from app.utils.llm import file_namer
import csv
import logging
from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from msgraph import GraphServiceClient
import pandas as pd
import json
import logging
from typing import Any
import os
from app.utils.auth import SupabaseClient
from azure.core.credentials import AccessToken
from datetime import datetime, timedelta, timezone
from azure.core.credentials import TokenCredential
from azure.identity import ClientSecretCredential, OnBehalfOfCredential
from msal import ConfidentialClientApplication
import aiohttp
import requests




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
    def __init__(self, google_refresh_token: str, microsoft_refresh_token: str = None):
        # Initialize _one_drive_id
        self._one_drive_id = None

        # Google credentials
        self.google_creds = Credentials(
            token=None,
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            refresh_token=google_refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            scopes=['https://www.googleapis.com/auth/drive.file']
        )

        # Microsoft auth setup
        if microsoft_refresh_token:
            try:
                # Microsoft OAuth2 token refresh request
                data = {
                    'client_id': os.getenv('MS_CLIENT_ID'),
                    'client_secret': os.getenv('MS_CLIENT_SECRET'),
                    'refresh_token': microsoft_refresh_token,
                    'grant_type': 'refresh_token',
                    'scope': 'https://graph.microsoft.com/.default'  # Changed to match working implementation
                }
                
                response = requests.post(
                    'https://login.microsoftonline.com/common/oauth2/v2.0/token',
                    data=data
                )
                response.raise_for_status()
                token_data = response.json()
                
                self.ms_access_token = token_data['access_token']
                self.ms_refresh_token = token_data.get('refresh_token', microsoft_refresh_token)
                self.ms_token_type = token_data['token_type']
                self.ms_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(token_data['expires_in']))
                
                logging.info("Successfully initialized Microsoft credentials")

            except Exception as e:
                logging.error(f"Failed to initialize Microsoft credentials: {str(e)}")
                raise

    async def _refresh_microsoft_token(self) -> None:
        """Refresh Microsoft OAuth token"""
        try:
            logging.info("Attempting to refresh Microsoft token")
            
            data = {
                'client_id': os.getenv('MS_CLIENT_ID'),
                'client_secret': os.getenv('MS_CLIENT_SECRET'),
                'refresh_token': self.ms_refresh_token,
                'grant_type': 'refresh_token',
                'scope': 'https://graph.microsoft.com/.default'  # Changed to match working implementation
            }
            
            response = requests.post(
                'https://login.microsoftonline.com/common/oauth2/v2.0/token',
                data=data
            )
            response.raise_for_status()
            token_data = response.json()
            
            self.ms_access_token = token_data['access_token']
            self.ms_refresh_token = token_data.get('refresh_token', self.ms_refresh_token)
            self.ms_token_type = token_data['token_type']
            self.ms_expires_at = datetime.now(timezone.utc) + timedelta(seconds=int(token_data['expires_in']))
            
            logging.info("Successfully refreshed Microsoft token")
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to refresh Microsoft token - HTTP error: {str(e)}")
            raise HTTPException(status_code=401, detail="Failed to refresh Microsoft token")
        except Exception as e:
            logging.error(f"Error refreshing Microsoft token: {str(e)}")
            raise

    async def _ensure_valid_microsoft_token(self) -> None:
        """Check and refresh Microsoft token if expired"""
        if datetime.now(timezone.utc) >= self.ms_expires_at:
            logging.info("Microsoft token is expired, refreshing...")
            await self._refresh_microsoft_token()

    async def _get_microsoft_headers(self) -> dict:
        """Get valid Microsoft API headers with current token"""
        await self._ensure_valid_microsoft_token()
        return {
            'Authorization': f"{self.ms_token_type} {self.ms_access_token}",
            'Content-Type': 'application/json'
        }

    def _format_data_for_excel(self, data: Any) -> List[List[Any]]:
        """Helper method to format data for Excel"""
        if isinstance(data, pd.DataFrame):
            df_copy = data.copy()
            for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            df_copy = df_copy.fillna('')
            return [df_copy.columns.tolist()] + df_copy.values.tolist()
        elif isinstance(data, (dict, list)):
            if isinstance(data, dict):
                processed_dict = {}
                for k, v in data.items():
                    if isinstance(v, pd.Timestamp):
                        processed_dict[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                    elif pd.isna(v):
                        processed_dict[k] = ''
                    else:
                        processed_dict[k] = str(v)
                return [[k, v] for k, v in processed_dict.items()]
            else:
                values = []
                for v in data:
                    if isinstance(v, pd.Timestamp):
                        values.append([v.strftime('%Y-%m-%d %H:%M:%S')])
                    elif pd.isna(v):
                        values.append([''])
                    else:
                        values.append([str(v)])
                return values
        return [[str(data)]]
    
    async def _get_one_drive_id(self) -> str:
        """Get the drive ID using the Graph API for personal OneDrive"""
        if not self._one_drive_id:
            try:
                logging.info("Attempting to get OneDrive ID...")

                headers = {
                    'Authorization': f'Bearer {self.ms_access_token}'
                }
                url = 'https://graph.microsoft.com/v1.0/me/drive'

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers) as response:
                        if response.status == 200:
                            drive_info = await response.json()
                            self._one_drive_id = drive_info['id']
                            logging.info(f"Successfully got OneDrive ID: {self._one_drive_id[:8]}...")
                        else:
                            response_text = await response.text()
                            logging.error(f"Failed to get OneDrive ID: {response_text}")
                            if "InvalidAuthenticationToken" in response_text:
                                raise HTTPException(
                                    status_code=401,
                                    detail="Invalid or expired Microsoft access token. Please reconnect your Microsoft account."
                                )
                            else:
                                raise HTTPException(
                                    status_code=response.status,
                                    detail=f"OneDrive access failed: {response_text}"
                                )
            except Exception as e:
                logging.error(f"Failed to get OneDrive ID: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to get OneDrive ID: {str(e)}"
                )

        return self._one_drive_id

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

    async def append_to_current_google_sheet(self, data: Any, sheet_url: str) -> bool:
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
                # Convert DataFrame to string values, handling Timestamps and NaN
                df_copy = data.copy()
                # Handle datetime columns
                for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                # Replace NaN values with empty string
                df_copy = df_copy.fillna('')
                values = df_copy.values.tolist()
            elif isinstance(data, (dict, list)):
                if isinstance(data, dict):
                    # Convert any Timestamp values to strings and handle NaN
                    processed_dict = {}
                    for k, v in data.items():
                        if isinstance(v, pd.Timestamp):
                            processed_dict[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        elif pd.isna(v):  # Handle NaN values
                            processed_dict[k] = ''
                        else:
                            processed_dict[k] = str(v)
                    values = [[k, v] for k, v in processed_dict.items()]
                else:
                    # Convert any Timestamp values in list to strings and handle NaN
                    values = []
                    for v in data:
                        if isinstance(v, pd.Timestamp):
                            values.append([v.strftime('%Y-%m-%d %H:%M:%S')])
                        elif pd.isna(v):  # Handle NaN values
                            values.append([''])
                        else:
                            values.append([str(v)])
            
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

    async def append_to_new_google_sheet(self, data: Any, sheet_url: str, old_data: List[FileDataInfo], query: str) -> bool:
        """Add data to a new sheet within an existing Google Sheets workbook"""
        try:
            # Create Google Sheets service
            service = build('sheets', 'v4', credentials=self.google_creds)
            
            # Extract spreadsheet ID from URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # Generate a unique sheet name (e.g. "Query Results 1", "Query Results 2", etc.)
            sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            existing_sheets = [sheet['properties']['title'] for sheet in sheet_metadata['sheets']]
            base_name = file_namer(query, old_data)
            sheet_name = base_name
            counter = 1
            while sheet_name in existing_sheets:
                sheet_name = f"{base_name} {counter}"
                counter += 1
            
            # Add new sheet
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': sheet_name
                        }
                    }
                }]
            }
            response = service.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body=body
            ).execute()
            
            # Format data for sheets (same as before)
            if isinstance(data, pd.DataFrame):
                df_copy = data.copy()
                for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_copy = df_copy.fillna('')
                values = [df_copy.columns.tolist()] + df_copy.values.tolist()
            elif isinstance(data, (dict, list)):
                if isinstance(data, dict):
                    processed_dict = {}
                    for k, v in data.items():
                        if isinstance(v, pd.Timestamp):
                            processed_dict[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        elif pd.isna(v):
                            processed_dict[k] = ''
                        else:
                            processed_dict[k] = str(v)
                    values = [[k, v] for k, v in processed_dict.items()]
                else:
                    values = []
                    for v in data:
                        if isinstance(v, pd.Timestamp):
                            values.append([v.strftime('%Y-%m-%d %H:%M:%S')])
                        elif pd.isna(v):
                            values.append([''])
                        else:
                            values.append([str(v)])
            else:
                values = [[str(data)]]
            
            # Write data to new sheet
            body = {
                'values': values
            }
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=f"{sheet_name}!A1",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            return True
            
        except Exception as e:
            logging.error(f"New Google Sheet creation error: {str(e)}")
            raise

    async def _get_drive_and_item_info(self, file_url: str) -> Tuple[str, str]:
        """Get drive ID and item ID from a OneDrive URL"""
        try:
            # Get drive ID
            drive_id = await self._get_one_drive_id()

            # Extract item ID from URL
            if 'id=' not in file_url:
                raise ValueError("Could not find item ID in OneDrive URL")

            # Parse out the item ID between id= and next & or end of string
            item_id = file_url.split('id=')[1].split('&')[0]
            
            if not item_id:
                raise ValueError("Empty item ID extracted from OneDrive URL")

            logging.info(f"Extracted item ID: {item_id}")
            return drive_id, item_id

        except Exception as e:
            logging.error(f"Failed to get drive and item info: {str(e)}")
            raise

    async def append_to_office_doc(self, data: Any, doc_url: str) -> bool:
        """Append data to Office Word Online document"""
        try:
            # Get drive and document info
            drive_id, doc_id = await self._get_drive_and_item_info(doc_url)

            # Format content
            if isinstance(data, pd.DataFrame):
                content = data.to_string()
            else:
                content = str(data)

            # Get valid headers with current token
            headers = await self._get_microsoft_headers()
            
            # Update document
            update_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{doc_id}/content"
            async with aiohttp.ClientSession() as session:
                async with session.put(update_url, headers=headers, data=content) as response:
                    if response.status in [200, 201, 204]:
                        logging.info("Successfully updated the Office document")
                        return True
                    else:
                        response_text = await response.text()
                        logging.error(f"Office Word append error: {response_text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to update Office document: {response_text}"
                        )

        except Exception as e:
            logging.error(f"Office Word append error: {str(e)}")
            raise

    async def append_to_current_office_sheet(self, data: Any, sheet_url: str) -> bool:
        """Append data to Office Excel Online workbook"""
        try:
            # Get drive ID
            drive_id = await self._get_one_drive_id()
            
            # Extract workbook ID from URL
            workbook_id = sheet_url.split('/')[-2]
            
            # Verify file exists and is Excel workbook
            item = await self.ms_client.drives.by_drive_id(drive_id).items.by_item_id(workbook_id).get()
            if not item.file or 'sheet' not in item.file.mime_type.lower():
                raise ValueError("Specified file is not an Excel workbook")
            
            # Format data for Excel
            values = self._format_data_for_excel(data)
            
            # Get the active worksheet
            workbook = await self.ms_client.drives.by_drive_id(drive_id).items.by_item_id(workbook_id).workbook.get()
            worksheets = await self.ms_client.drives.by_drive_id(drive_id).items.by_item_id(workbook_id).workbook.worksheets.get()
            active_sheet = worksheets.value[0]  # Get first worksheet
            
            # Find the next empty row
            used_range = await self.ms_client.drives.by_drive_id(drive_id).items.by_item_id(workbook_id).workbook.worksheets.by_id(active_sheet.id).used_range.get()
            next_row = used_range.row_count + 1
            
            # Write data to worksheet
            range_address = f"A{next_row}:${chr(65 + len(values[0]) - 1)}${next_row + len(values) - 1}"
            await self.ms_client.drives.by_drive_id(drive_id).items.by_item_id(workbook_id).workbook.worksheets.by_id(active_sheet.id).range(range_address).values.set(values)
            
            return True
            
        except Exception as e:
            logging.error(f"Office Excel append error: {str(e)}")
            raise

    async def append_to_new_office_sheet(self, data: Any, sheet_url: str, old_data: List[FileDataInfo], query: str) -> bool:
        """Add data to a new sheet within an existing Office Excel workbook"""
        try:
            # Initialize Microsoft Graph client
            graph_client = GraphServiceClient(self.ms_access_token)
            
            # Extract workbook ID from URL
            workbook_id = sheet_url.split('/')[-2]
            
            # Generate a unique sheet name
            workbook = await graph_client.me.drive.items[workbook_id].workbook.get()
            worksheets = await graph_client.me.drive.items[workbook_id].workbook.worksheets.get()
            existing_sheets = [ws.name for ws in worksheets]
            
            base_name = file_namer(query, old_data)
            sheet_name = base_name
            counter = 1
            while sheet_name in existing_sheets:
                sheet_name = f"{base_name} {counter}"
                counter += 1
            
            # Create new worksheet in existing workbook
            worksheet = await graph_client.me.drive.items[workbook_id].workbook.worksheets.add(sheet_name)
            
            # Format data for Excel (same as before)
            if isinstance(data, pd.DataFrame):
                df_copy = data.copy()
                for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_copy = df_copy.fillna('')
                values = [df_copy.columns.tolist()] + df_copy.values.tolist()
            elif isinstance(data, (dict, list)):
                if isinstance(data, dict):
                    processed_dict = {}
                    for k, v in data.items():
                        if isinstance(v, pd.Timestamp):
                            processed_dict[k] = v.strftime('%Y-%m-%d %H:%M:%S')
                        elif pd.isna(v):
                            processed_dict[k] = ''
                        else:
                            processed_dict[k] = str(v)
                    values = [[k, v] for k, v in processed_dict.items()]
                else:
                    values = []
                    for v in data:
                        if isinstance(v, pd.Timestamp):
                            values.append([v.strftime('%Y-%m-%d %H:%M:%S')])
                        elif pd.isna(v):
                            values.append([''])
                        else:
                            values.append([str(v)])
            else:
                values = [[str(data)]]
            
            # Write data to new worksheet
            range_address = f"A1:${chr(65 + len(values[0]) - 1)}${len(values)}"
            await worksheet.range(range_address).values.set(values)
            
            return True
            
        except Exception as e:
            logging.error(f"New Office Excel worksheet creation error: {str(e)}")
            raise

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
        
        g_response = supabase.table('user_documents_access') \
            .select('refresh_token') \
            .match({'user_id': user_id, 'provider': 'google'}) \
            .execute()
        
        if not g_response.data or len(g_response.data) == 0:
            print(f"No Google token found for user {user_id}")
            return None
        google_refresh_token = g_response.data[0]['refresh_token']

        ms_response = supabase.table('user_documents_access') \
            .select('refresh_token') \
            .match({'user_id': user_id, 'provider': 'microsoft'}) \
            .execute()
        
        if not ms_response.data or len(ms_response.data) == 0:
            print(f"No Microsoft token found for user {user_id}")
            return None
        ms_refresh_token = ms_response.data[0]['refresh_token']


        doc_integrations = DocumentIntegrations(google_refresh_token, ms_refresh_token)
        url_lower = request.output_preferences.destination_url.lower()
        
        if "docs.google.com" in url_lower:
            if "document" in url_lower:
                return await doc_integrations.append_to_google_doc(data, request.output_preferences.destination_url)
            elif "spreadsheets" in url_lower:
                if request.output_preferences.modify_existing:
                    return await doc_integrations.append_to_current_google_sheet(data, request.output_preferences.destination_url)
                else:
                    return await doc_integrations.append_to_new_google_sheet(data, request.output_preferences.destination_url, old_data, request.query)
        
        elif "onedrive" in url_lower or "sharepoint.com" in url_lower:
            if "docx" in url_lower:
                return await doc_integrations.append_to_office_doc(data, request.output_preferences.destination_url)
            elif "xlsx" in url_lower:
                if request.output_preferences.modify_existing:
                    return await doc_integrations.append_to_current_office_sheet(data, request.output_preferences.destination_url)
                else:
                    return await doc_integrations.append_to_new_office_sheet(data, request.output_preferences.destination_url, old_data, request.query)
        
        raise ValueError(f"Unsupported destination URL type: {request.output_preferences.destination_url}")
    
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