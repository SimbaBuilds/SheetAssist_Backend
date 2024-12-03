from app.schemas import FileDataInfo
from typing import List, Tuple
from app.utils.llm import file_namer
import logging
from fastapi import HTTPException
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from msgraph import GraphServiceClient
import pandas as pd
import os
from datetime import datetime, timedelta, timezone
import aiohttp
import requests
from typing import Any
from pydantic import BaseModel
from azure.core.credentials import AccessToken
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient

# Define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Add TokenInfo model
class TokenInfo(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str
    scope: str
    expires_at: str
    user_id: str

class DocumentIntegrations:
    def __init__(self, google_refresh_token: str, microsoft_refresh_token: str = None, user_id: str = None, supabase_client = None):
        self._one_drive_id = None
        self.user_id = user_id
        self.supabase_client = supabase_client

        # Google credentials
        self.google_creds = Credentials(
            token=None,
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            refresh_token=google_refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            scopes=['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
        )

        # Microsoft auth setup
        if microsoft_refresh_token:
            self.token_info = TokenInfo(
                access_token="",  # Will be set in _refresh_microsoft_token
                refresh_token=microsoft_refresh_token,
                token_type="",    # Will be set in _refresh_microsoft_token
                scope="https://graph.microsoft.com/.default",
                expires_at=datetime.now(timezone.utc).isoformat(),
                user_id=user_id
            )
            self._refresh_microsoft_token()
            
            # Create auth callback
            def auth_callback(request):
                request.headers.update({
                    'Authorization': f'Bearer {self.token_info.access_token}',
                    'Content-Type': 'application/json'
                })
                return request

            # Initialize Microsoft Graph client
            self.ms_client = GraphClient(
                auth_provider=AuthProviderCallback(auth_callback),
                base_url="https://graph.microsoft.com/v1.0"
            )

    async def _refresh_microsoft_token(self) -> None:
        """Refresh Microsoft OAuth token and update in database"""
        try:
            logger.info("Attempting to refresh Microsoft token")
            
            # Microsoft OAuth2 token refresh request
            data = {
                'client_id': os.getenv('MS_CLIENT_ID'),
                'client_secret': os.getenv('MS_CLIENT_SECRET'),
                'refresh_token': self.token_info.refresh_token,
                'grant_type': 'refresh_token'
            }
            
            response = requests.post(
                'https://login.microsoftonline.com/common/oauth2/v2.0/token',
                data=data
            )
            response.raise_for_status()
            token_data = response.json()
            logger.info(f"Received token data: {token_data}")
            
            # Update token info
            self.token_info.access_token = token_data['access_token']
            self.token_info.refresh_token = token_data.get('refresh_token', self.token_info.refresh_token)
            self.token_info.token_type = token_data['token_type']
            self.token_info.scope = token_data['scope']
            self.token_info.expires_at = (datetime.now(timezone.utc) + 
                                        timedelta(seconds=int(token_data['expires_in']))
                                       ).isoformat()
            
            # Update the Graph client with new credentials
            self.ms_client = GraphServiceClient(
                credentials=self.token_info.access_token,
                scopes=[self.token_info.scope]
            )
            
            # Update token in database if supabase client is available
            if self.supabase_client and self.user_id:
                response = self.supabase_client.table('user_documents_access') \
                    .update({
                        'access_token': self.token_info.access_token,
                        'refresh_token': self.token_info.refresh_token,
                        'expires_at': self.token_info.expires_at
                    }) \
                    .match({
                        'provider': 'microsoft',
                        'user_id': self.user_id
                    }) \
                    .execute()
                logger.info("Successfully updated Microsoft token in database")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to refresh Microsoft token - HTTP error: {str(e)}")
            raise HTTPException(status_code=401, detail="Failed to refresh Microsoft token")
        except Exception as e:
            logger.error(f"Error refreshing Microsoft token: {str(e)}")
            raise HTTPException(status_code=401, detail="Failed to refresh Microsoft token")

    async def _ensure_valid_microsoft_token(self) -> None:
        """Check and refresh Microsoft token if expired"""
        current_time = datetime.now(timezone.utc)
        token_expires_at = datetime.fromisoformat(self.token_info.expires_at.replace('Z', '+00:00'))
        if current_time >= token_expires_at:
            await self._refresh_microsoft_token()

    async def _get_microsoft_headers(self) -> dict:
        """Get valid Microsoft API headers with current token"""
        await self._ensure_valid_microsoft_token()
        return {
            'Authorization': f"{self.token_info.token_type} {self.token_info.access_token}",
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
            headers = await self._get_microsoft_headers()
            async with aiohttp.ClientSession() as session:
                async with session.get('https://graph.microsoft.com/v1.0/me/drive', headers=headers) as response:
                    if response.status == 200:
                        drive_info = await response.json()
                        self._one_drive_id = drive_info['id']
                    else:
                        response_text = await response.text()
                        if "InvalidAuthenticationToken" in response_text:
                            raise HTTPException(status_code=401, detail="Invalid or expired Microsoft access token")
                        raise HTTPException(status_code=response.status, detail=f"OneDrive access failed: {response_text}")
        return self._one_drive_id

    async def _get_one_drive_and_item_info(self, file_url: str) -> Tuple[str, str]:
        """Get drive ID and item ID from a OneDrive URL"""
        drive_id = await self._get_one_drive_id()
        if 'id=' not in file_url:
            raise ValueError("Could not find item ID in OneDrive URL")
        item_id = file_url.split('id=')[1].split('&')[0]
        if not item_id:
            raise ValueError("Empty item ID extracted from OneDrive URL")
        return drive_id, item_id

    async def _manage_office_session(self, item_id: str, action: str = 'create') -> str:
        """Manage Microsoft Office Excel sessions"""
        headers = await self._get_microsoft_headers()
        base_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/workbook'
        
        if action == 'create':
            async with aiohttp.ClientSession() as session:
                async with session.post(f'{base_url}/createSession', headers=headers, json={"persistChanges": True}) as response:
                    if response.status == 201:
                        response_data = await response.json()
                        return response_data['id']
                    response_text = await response.text()
                    raise HTTPException(status_code=response.status, detail=f"Failed to create Office session: {response_text}")
        elif action == 'list':
            async with aiohttp.ClientSession() as session:
                async with session.get(f'{base_url}/sessions', headers=headers) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return response_data.get('value', [])
                    return []
        elif action == 'close' and item_id:
            async with aiohttp.ClientSession() as session:
                async with session.delete(f'{base_url}/sessions/{item_id}', headers=headers) as response:
                    return response.status in [200, 204]
        return None

    #MSFT INTEGRATIONS

    async def append_to_current_office_sheet(self, data: Any, sheet_url: str) -> bool:
        """Append data to Office Excel Online workbook"""
        try:
            drive_id, workbook_id = await self._get_one_drive_and_item_info(sheet_url)
            
            # Get worksheets
            response = await self.ms_client.get(
                f'/drives/{drive_id}/items/{workbook_id}/workbook/worksheets'
            )
            worksheets = response.json()['value']
            active_sheet = worksheets[0]  # Get first worksheet
            
            # Get used range
            response = await self.ms_client.get(
                f'/drives/{drive_id}/items/{workbook_id}/workbook/worksheets/{active_sheet["id"]}/usedRange'
            )
            used_range = response.json()
            next_row = used_range['rowCount'] + 1
            
            # Format data
            values = self._format_data_for_excel(data)
            
            # Write data
            range_address = f"A{next_row}:${chr(65 + len(values[0]) - 1)}${next_row + len(values) - 1}"
            await self.ms_client.patch(
                f'/drives/{drive_id}/items/{workbook_id}/workbook/worksheets/{active_sheet["id"]}/range(address=\'{range_address}\')',
                json={'values': values}
            )
            
            return True
            
        except Exception as e:
            logging.error(f"Office Excel append error: {str(e)}")
            raise

    async def append_to_new_office_sheet(self, data: Any, sheet_url: str, old_data: List[FileDataInfo], query: str) -> bool:
        """Add data to a new sheet within an existing Office Excel workbook"""
        try:
            drive_id, workbook_id = await self._get_one_drive_and_item_info(sheet_url)
            
            # Get existing worksheets
            response = await self.ms_client.get(
                f'/drives/{drive_id}/items/{workbook_id}/workbook/worksheets'
            )
            existing_sheets = [ws['name'] for ws in response.json()['value']]
            
            # Generate unique sheet name
            base_name = file_namer(query, old_data)
            sheet_name = base_name
            counter = 1
            while sheet_name in existing_sheets:
                sheet_name = f"{base_name} {counter}"
                counter += 1
            
            # Create new worksheet
            response = await self.ms_client.post(
                f'/drives/{drive_id}/items/{workbook_id}/workbook/worksheets/add',
                json={'name': sheet_name}
            )
            new_sheet = response.json()
            
            # Format and write data
            values = self._format_data_for_excel(data)
            range_address = f"A1:${chr(65 + len(values[0]) - 1)}${len(values)}"
            
            await self.ms_client.patch(
                f'/drives/{drive_id}/items/{workbook_id}/workbook/worksheets/{new_sheet["id"]}/range(address=\'{range_address}\')',
                json={'values': values}
            )
            
            return True
            
        except Exception as e:
            logging.error(f"New Office Excel worksheet creation error: {str(e)}")
            raise

    #GOOGLE INTEGRATIONS

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
