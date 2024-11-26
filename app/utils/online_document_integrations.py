from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
import pandas as pd
import json
import logging
from typing import Any
import os

class DocumentIntegrations:
    def __init__(self):
        # Google credentials
        self.google_creds = Credentials.from_authorized_user_info(
            json.loads(os.getenv('GOOGLE_CREDENTIALS')),
            ['https://www.googleapis.com/auth/documents',
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
            
            # Create Google Sheets service
            service = build('sheets', 'v4', credentials=self.google_creds)
            
            # Create new sheet
            sheet_title = "Query Results"
            body = {
                'requests': [{
                    'addSheet': {
                        'properties': {
                            'title': sheet_title
                        }
                    }
                }]
            }
            service.spreadsheets().batchUpdate(
                spreadsheetId=sheet_id,
                body=body
            ).execute()
            
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
            
            # Append data to new sheet
            body = {
                'values': values
            }
            service.spreadsheets().values().update(
                spreadsheetId=sheet_id,
                range=f"{sheet_title}!A1",
                valueInputOption='RAW',
                body=body
            ).execute()
            
            return True
            
        except Exception as e:
            logging.error(f"Google Sheets append error: {str(e)}")
            raise

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