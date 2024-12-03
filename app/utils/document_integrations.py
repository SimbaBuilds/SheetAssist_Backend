from app.schemas import FileDataInfo
from typing import List, Tuple
from app.utils.llm import file_namer
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
from datetime import datetime, timedelta, timezone
import aiohttp
import requests
from app.utils.file_management import temp_file_manager
from fastapi import BackgroundTasks
import aiofiles



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

    async def _get_one_drive_and_item_info(self, file_url: str) -> Tuple[str, str]:
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

    async def _get_office_sessions(self, item_id: str, is_excel: bool = True) -> list:
        """Get all active sessions for a Microsoft Office Excel file"""
        if not is_excel:
            return []
            
        try:
            headers = await self._get_microsoft_headers()
            list_sessions_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/workbook/sessions'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(list_sessions_url, headers=headers) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        return response_data.get('value', [])
                    else:
                        response_text = await response.text()
                        logging.error(f"Failed to get Office sessions: {response_text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to get Office sessions: {response_text}"
                        )
        except Exception as e:
            logging.error(f"Error getting Office sessions: {str(e)}")
            raise

    async def _close_office_session(self, item_id: str, session_id: str, is_excel: bool = True) -> None:
        """Close a specific Microsoft Office Excel session"""
        if not is_excel:
            return
            
        try:
            headers = await self._get_microsoft_headers()
            delete_session_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/workbook/sessions/{session_id}'
            
            async with aiohttp.ClientSession() as session:
                async with session.delete(delete_session_url, headers=headers) as response:
                    if response.status not in [200, 204]:
                        response_text = await response.text()
                        logging.error(f"Failed to close Office session: {response_text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to close Office session: {response_text}"
                        )
        except Exception as e:
            logging.error(f"Error closing Office session: {str(e)}")
            raise

    async def _create_office_session(self, item_id: str, is_excel: bool = True) -> str:
        """Create a new Microsoft Office Excel session"""
        if not is_excel:
            return None
            
        try:
            headers = await self._get_microsoft_headers()
            create_session_url = f'https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/workbook/createSession'
            session_body = {
                "persistChanges": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(create_session_url, headers=headers, json=session_body) as response:
                    if response.status == 201:
                        response_data = await response.json()
                        return response_data['id']
                    else:
                        response_text = await response.text()
                        logging.error(f"Failed to create Office session: {response_text}")
                        raise HTTPException(
                            status_code=response.status,
                            detail=f"Failed to create Office session: {response_text}"
                        )
        except Exception as e:
            logging.error(f"Error creating Office session: {str(e)}")
            raise

    #MSFT INTEGRATIONS


    async def append_to_current_office_sheet(self, data: Any, sheet_url: str) -> bool:
        """Append data to Office Excel Online workbook"""
        try:
            # Get drive ID and workbook ID
            drive_id, workbook_id = await self._get_one_drive_and_item_info(sheet_url)
            
            # Get active sessions and close them
            sessions = await self._get_office_sessions(workbook_id, is_excel=True)
            for session in sessions:
                await self._close_office_session(workbook_id, session['id'], is_excel=True)

            # Create new session
            session_id = await self._create_office_session(workbook_id, is_excel=True)

            try:
                # Get valid headers with current token
                headers = await self._get_microsoft_headers()
                headers['workbook-session-id'] = session_id

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
            finally:
                # Always close the session after we're done
                if session_id:
                    await self._close_office_session(workbook_id, session_id, is_excel=True)
            
        except Exception as e:
            logging.error(f"Office Excel append error: {str(e)}")
            raise

    async def append_to_new_office_sheet(self, data: Any, sheet_url: str, old_data: List[FileDataInfo], query: str) -> bool:
        """Add data to a new sheet within an existing Office Excel workbook"""
        try:
            # Get drive ID and workbook ID
            drive_id, workbook_id = await self._get_one_drive_and_item_info(sheet_url)
            
            # Get active sessions and close them
            sessions = await self._get_office_sessions(workbook_id, is_excel=True)
            for session in sessions:
                await self._close_office_session(workbook_id, session['id'], is_excel=True)

            # Create new session
            session_id = await self._create_office_session(workbook_id, is_excel=True)

            try:
                # Get valid headers with current token
                headers = await self._get_microsoft_headers()
                if session_id:
                    headers['workbook-session-id'] = session_id

                # Initialize Microsoft Graph client
                graph_client = GraphServiceClient(self.ms_access_token)
                
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
                
                # Format data for Excel
                values = self._format_data_for_excel(data)
                
                # Write data to new worksheet
                range_address = f"A1:${chr(65 + len(values[0]) - 1)}${len(values)}"
                await worksheet.range(range_address).values.set(values)
                
                return True
            finally:
                # Always close the session after we're done
                if session_id:
                    await self._close_office_session(workbook_id, session_id, is_excel=True)
            
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
