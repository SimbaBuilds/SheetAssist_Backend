from app.schemas import FileDataInfo
import logging
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import pandas as pd
import os
from datetime import datetime
from typing import Any
from supabase.client import Client as SupabaseClient
from datetime import date
from typing import List


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
            scopes =  [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file',
    'https://www.googleapis.com/auth/drive.readonly',
]
        )

    def _format_data_for_sheets(self, data: Any) -> List[List[str]]:
        """Helper function to format data for Google Sheets."""
        def format_value(v: Any) -> str:
            """Helper function to format individual values"""
            # Handle pandas Series
            if isinstance(v, pd.Series):
                return format_value(v.iloc[0] if len(v) > 0 else '')
            # Handle other types
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
                df_copy.loc[:, col] = df_copy[col].apply(format_value)
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

    async def append_to_current_google_sheet(self, data: Any, sheet_url: str, sheet_name: str) -> bool:
        """Append data to Google Sheet"""
        try:
            # Extract spreadsheet ID from URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # Create Google Sheets service with proper scopes
            service = build('sheets', 'v4', credentials=self.google_creds)
            
            # Use the provided sheet_name directly
            if not sheet_name:
                # Fallback to first sheet if no sheet name provided
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
                range=f"{sheet_name}",
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            return True
            
        except Exception as e:
            logging.error(f"Google Sheets append error: {str(e)}")
            raise

    async def append_to_new_google_sheet(self, data: Any, sheet_url: str, sheet_name: str) -> bool:
        """Add data to a new sheet within an existing Google Sheets workbook"""
        try:
            # Extract spreadsheet ID from URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # Create Google Sheets service
            service = build('sheets', 'v4', credentials=self.google_creds)
            
            # Get existing sheets
            sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            existing_sheets = [sheet['properties']['title'] for sheet in sheet_metadata['sheets']]
            
            # Ensure unique sheet name
            final_name = sheet_name
            counter = 1
            while final_name in existing_sheets:
                final_name = f"{sheet_name} {counter}"
                counter += 1
            
            # Add new sheet
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': final_name
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
                range=f"{final_name}!A1",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            return True
            
        except Exception as e:
            logging.error(f"New Google Sheet creation error: {str(e)}")
            raise

#preprocessing
    async def extract_google_sheets_data(self, sheet_url: str, sheet_name: str) -> pd.DataFrame:
        """
        Extract data from a Google Sheets URL using the provided sheet name.
        
        Args:
            sheet_url (str): URL of the Google Sheet
            sheet_name (str): Name of the sheet to extract data from
            
        Returns:
            pd.DataFrame: DataFrame containing the sheet data
            
        Raises:
            ValueError: If URL is invalid or file cannot be accessed
        """
        if 'docs.google.com/spreadsheets' not in sheet_url:
            raise ValueError("Invalid Google Sheets URL format")
            
        try:
            # Extract file ID from URL
            file_id = sheet_url.split('/d/')[1].split('/')[0]
            
            # Create range with sheet name
            sheet_range = f"'{sheet_name}'!A:ZZ"
            
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