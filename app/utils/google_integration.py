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
from supabase.client import Client as SupabaseClient
from datetime import date


class GoogleIntegration:
    def __init__(self, supabase: SupabaseClient, user_id: str):
        self.google_creds = None

        g_response = supabase.table('user_documents_access') \
        .select('refresh_token') \
        .match({'user_id': user_id, 'provider': 'google'}) \
        .execute()
        
        if not g_response.data or len(g_response.data) == 0:
            print(f"No Google token found for user {user_id}")
            return None
        google_refresh_token = g_response.data[0]['refresh_token']
        
        # Google credentials
        self.google_creds = Credentials(
            token=None,
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            refresh_token=google_refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            scopes=['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/spreadsheets']
        )

    def _format_data_for_sheets(self, data: Any) -> List[List[str]]:
        """Helper function to format data for Google Sheets.
        
        Args:
            data: Data to format (DataFrame, dict, list, or scalar value)
            
        Returns:
            List of lists containing formatted string values
        """
        def format_value(v: Any) -> str:
            """Helper function to format individual values"""
            if isinstance(v, pd.Timestamp):
                return v.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(v, (datetime, date)):
                return v.strftime('%Y-%m-%d')
            elif pd.isna(v):
                return ''
            else:
                return str(v)

        if isinstance(data, pd.DataFrame):
            df_copy = data.copy()
            # Handle datetime columns
            for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
            # Replace NaN values with empty string
            df_copy = df_copy.fillna('')
            # Convert any remaining date objects
            for col in df_copy.columns:
                if df_copy[col].dtype == 'object':
                    df_copy[col] = df_copy[col].apply(format_value)
            return df_copy.values.tolist()
        elif isinstance(data, dict):
            # Convert any Timestamp values to strings and handle NaN
            processed_dict = {k: format_value(v) for k, v in data.items()}
            return [[k, v] for k, v in processed_dict.items()]
        elif isinstance(data, list):
            # Convert any Timestamp values in list to strings and handle NaN
            return [[format_value(v)] for v in data]
        else:
            return [[format_value(data)]]

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
            
            # Format data for sheets using helper function
            values = self._format_data_for_sheets(data)
            
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
            
            # Format data for sheets using helper function
            values = self._format_data_for_sheets(data)
            if isinstance(data, pd.DataFrame):
                # Add column headers for DataFrame
                values = [data.columns.tolist()] + values
            
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

#preprocessing
    async def extract_google_sheets_data(self, sheet_url: str) -> pd.DataFrame:
        """
        Extract data from a Google Sheets URL. The sheet name/id can be parsed from the URL if present.
        
        Args:
            sheet_url (str): URL of the Google Sheet
            
        Returns:
            pd.DataFrame: DataFrame containing the sheet data
            
        Raises:
            ValueError: If URL is invalid or file cannot be accessed
        """
        if 'docs.google.com/spreadsheets' not in url:
            raise ValueError("Invalid Google Sheets URL format")
            
        try:
            # Extract file ID from URL
            file_id = url.split('/d/')[1].split('/')[0]
            
            # Get the sheet name/gid from URL if present
            sheet_range = 'A:ZZ'  # Default range
            if '#gid=' in url:
                gid = url.split('#gid=')[1].split('&')[0]
                # Get sheet name from gid
                service = build('sheets', 'v4', credentials=self.google_creds)
                sheet_metadata = service.spreadsheets().get(spreadsheetId=file_id).execute()
                for sheet in sheet_metadata.get('sheets', []):
                    if sheet.get('properties', {}).get('sheetId') == int(gid):
                        sheet_name = sheet['properties']['title']
                        sheet_range = f"'{sheet_name}'!A:ZZ"
                        break
            
            # Use Google integration to get authenticated service
            service = build('sheets', 'v4', credentials=self.google_creds)
            if not service:
                raise ValueError("Failed to initialize Google Sheets service")
            
            # Read the sheet data using authenticated service
            result = service.spreadsheets().values().get(
                spreadsheetId=file_id,
                range=sheet_range
            ).execute()
            
            # Convert to DataFrame
            values = result.get('values', [])
            if not values:
                return pd.DataFrame()
            
            headers = values[0]
            data = values[1:]
            return pd.DataFrame(data, columns=headers)
            
        except Exception as e:
            error_msg = str(e)
            if any(sensitive in error_msg.lower() for sensitive in ['token', 'key', 'auth', 'password', 'secret']):
                error_msg = "Authentication error occurred while accessing the file"
            raise ValueError(error_msg)