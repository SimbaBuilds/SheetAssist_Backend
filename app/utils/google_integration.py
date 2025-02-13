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
import logging

logger = logging.getLogger(__name__)

class GoogleIntegration:
    def __init__(self, supabase: SupabaseClient = None, user_id: str = None, picker_token: str = None):
        self.google_creds = None

        if picker_token:
            logger.info("Using picker token for Google integration")
            # Use picker token directly
            self.google_creds = Credentials(
                token=picker_token,
                client_id=os.getenv('GOOGLE_CLIENT_ID'),
                client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
                token_uri='https://oauth2.googleapis.com/token',
                scopes=['https://www.googleapis.com/auth/drive.file']
            )
        else:
            logger.info("Using db stored token for Google integration")
            # Fallback to stored token if picker token not provided
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
                scopes=['https://www.googleapis.com/auth/drive.file']
            )

    def _format_data_for_sheets(self, data: Any) -> List[List[str]]:
        """Helper function to format data for Google Sheets."""
        if isinstance(data, pd.DataFrame):
            # Ensure the DataFrame is properly converted to a 2D list
            headers = [[str(col) for col in data.columns]]
            # Convert all values to strings and handle None/NaN
            values = [['' if pd.isna(x) else str(x) for x in row] for row in data.values]
            return headers + values
        
        # Handle dict and list cases
        if isinstance(data, dict):
            return [[str(k), str(v) if not pd.isna(v) else ''] for k, v in data.items()]
        elif isinstance(data, list):
            return [[str(v) if not pd.isna(v) else ''] for v in data]
        
        # Handle single value
        return [[str(data) if not pd.isna(data) else '']]

    async def append_to_current_google_sheet(self, data: Any, sheet_url: str, sheet_name: str) -> bool:
        """Append data to Google Sheet, avoiding duplicates"""
        logging.info(f"Starting append to Google Sheet: {sheet_name}")
        try:
            # Extract spreadsheet ID from URL
            sheet_id = sheet_url.split('/d/')[1].split('/')[0]
            logging.info(f"Extracted sheet ID: {sheet_id}")
            
            # Create Google Sheets service with proper scopes
            service = build('sheets', 'v4', credentials=self.google_creds)
            logging.info("Created Google Sheets service")
            
            # First get existing data to determine row count
            logging.info(f"Fetching existing data from sheet: {sheet_name}")
            result = service.spreadsheets().values().get(
                spreadsheetId=sheet_id,
                range=f"{sheet_name}"
            ).execute()
            
            existing_values = result.get('values', [])
            existing_row_count = len(existing_values)
            logging.info(f"Existing row count: {existing_row_count}")
            
            # Convert input data to DataFrame if it isn't already
            logging.info("Converting input data to DataFrame")
            if not isinstance(data, pd.DataFrame):
                if isinstance(data, dict):
                    data = pd.DataFrame([data])
                elif isinstance(data, list):
                    data = pd.DataFrame(data)
                else:
                    data = pd.DataFrame([data])
            logging.info(f"Data shape: {data.shape}")
            # Get only the new rows (rows beyond existing count)
            new_rows = data.iloc[existing_row_count-1:]
            logging.info(f"New rows to append: {len(new_rows)}")
            print(f"New rows to append: {len(new_rows)}")
            all_new = False
            if len(new_rows) == 0:
                logging.info("No new rows detected, processing all passed data")
                all_new = True
                new_rows = data  # Use all passed data instead of returning
            
            # Format the new data for sheets
            logging.info("Formatting data for Google Sheets")
            values = self._format_data_for_sheets(new_rows)
            # Remove header row since we're appending to existing sheet
            if not all_new:
                values = values[1:] if values else []
            logging.info(f"Formatted {len(values)} rows")
            
            # Append the data
            body = {
                'values': values
            }
            
            logging.info("Appending data to sheet")
            append_result = service.spreadsheets().values().append(
                spreadsheetId=sheet_id,
                range=f"{sheet_name}!A1",  # Specify starting cell
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()
            
            updated_rows = append_result.get('updates', {}).get('updatedRows', 0)
            logging.info(f"Successfully appended {updated_rows} rows")
            logging.info(f"Full append result: {append_result}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to append to Google Sheet: {str(e)}", exc_info=True)
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
            
            # Ensure values is a 2D array of strings
            if not values or not isinstance(values[0], list):
                values = [[str(x) for x in values]]

            # Write data to new sheet
            body = {
                'values': values
            }
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=f"{final_name}!A1",
                valueInputOption='USER_ENTERED',  # Changed from 'RAW'
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

            # Create DataFrame with dynamic columns based on actual data
            df = pd.DataFrame(data)
            
            # If we have more headers than data columns, pad the data with empty strings
            if len(headers) > len(df.columns):
                for i in range(len(df.columns), len(headers)):
                    df[i] = ''  # Use empty string instead of pd.NA
            # If we have more data columns than headers, add generic headers
            elif len(headers) < len(df.columns):
                for i in range(len(headers), len(df.columns)):
                    headers.append(f'Column_{i+1}')
                
            df.columns = headers
            return df
            
        except Exception as e:
            error_msg = str(e)
            if any(sensitive in error_msg.lower() for sensitive in ['token', 'key', 'auth', 'password', 'secret']):
                error_msg = "Authentication error occurred while accessing the file"
            raise ValueError(error_msg)