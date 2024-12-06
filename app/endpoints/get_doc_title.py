from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
import requests
import os
from datetime import datetime, timezone, timedelta
from typing import Annotated
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client
import logging

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

class DocumentTitleRequest(BaseModel):
    url: str

class DocumentTitle(BaseModel):
    url: str
    title: str

class TokenInfo(BaseModel):
    access_token: str
    refresh_token: str
    expires_at: str
    token_type: str
    scope: str
    user_id: str

class WorkbookResponse(BaseModel):
    url: str
    doc_name: Optional[str] = None
    provider: Optional[str] = None
    sheet_names: Optional[list] = None
    error: Optional[str] = None
    success: bool

class OnlineSheet(BaseModel):
    doc_name: str
    provider: str
    sheet_names: list

async def get_provider_token(user_id: str, provider: str, supabase_client) -> Optional[TokenInfo]:
    """Fetch token for a specific provider from user_documents_access table"""
    try:
        logger.info(f"Fetching {provider} token for user {user_id}")
        response = supabase_client.table('user_documents_access') \
            .select('*') \
            .match({'user_id': user_id, 'provider': provider}) \
            .execute()
        
        if not response.data or len(response.data) == 0:
            logger.warning(f"No {provider} token found for user {user_id}")
            return None
            
        token_data = response.data[0]
        logger.info(f"Successfully retrieved {provider} token from database")
        return TokenInfo(**token_data)
    except Exception as e:
        logger.error(f"Error fetching {provider} token: {str(e)}")
        return None

async def refresh_google_token(token_info: TokenInfo, supabase_client) -> Optional[TokenInfo]:
    """Refresh Google OAuth token and update in database"""
    try:
        logger.info("Attempting to refresh Google token")
        creds = Credentials(
            token=token_info.access_token,
            refresh_token=token_info.refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            scopes=token_info.scope.split(' ')
        )
        
        # Refresh the credentials
        creds.refresh(Request())
        logger.info(f"Refreshed Google token: {creds.token}")
        
        # Calculate new expiry time
        expiry_timestamp = int(creds.expiry.timestamp())
        current_timestamp = int(datetime.now().timestamp())
        expires_in = expiry_timestamp - current_timestamp
        
        # Create new token info
        new_token_info = TokenInfo(
            access_token=creds.token,
            refresh_token=creds.refresh_token or token_info.refresh_token,
            token_type='Bearer',
            scope=' '.join(creds.scopes),
            expires_at=(datetime.now(timezone.utc) + 
                       timedelta(seconds=expires_in)
                      ).isoformat(),
            user_id=token_info.user_id
        )
        
        # Update token in database
        response = supabase_client.table('user_documents_access') \
            .update({
                'access_token': new_token_info.access_token,
                'refresh_token': new_token_info.refresh_token,
                'expires_at': new_token_info.expires_at
            }) \
            .match({
                'provider': 'google',
                'user_id': token_info.user_id
            }) \
            .execute()
            
        logger.info("Successfully updated Google token in database")
        return new_token_info
        
    except RefreshError as e:
        logger.error(f"Failed to refresh token - token may be revoked: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error refreshing Google token: {str(e)}")
        return None

async def refresh_microsoft_token(token_info: TokenInfo, supabase_client) -> Optional[TokenInfo]:
    """Refresh Microsoft OAuth token and update in database"""
    try:
        logger.info("Attempting to refresh Microsoft token")
        
        # Microsoft OAuth2 token refresh request
        data = {
            'client_id': os.getenv('MS_CLIENT_ID'),
            'client_secret': os.getenv('MS_CLIENT_SECRET'),
            'refresh_token': token_info.refresh_token,
            'grant_type': 'refresh_token'
        }
        
        response = requests.post(
            'https://login.microsoftonline.com/common/oauth2/v2.0/token',
            data=data
        )
        response.raise_for_status()
        token_data = response.json()
        logger.info(f"Received token data: {token_data}")
        # Create new token info
        new_token_info = TokenInfo(
            access_token=token_data['access_token'],
            refresh_token=token_data.get('refresh_token', token_info.refresh_token),
            token_type=token_data['token_type'],
            scope=token_data['scope'],
            expires_at=(datetime.now(timezone.utc) + 
                       timedelta(seconds=int(token_data['expires_in']))
                      ).isoformat(),
            user_id=token_info.user_id
        )
        
        # Update token in database
        response = supabase_client.table('user_documents_access') \
            .update({
                'access_token': new_token_info.access_token,
                'refresh_token': new_token_info.refresh_token,
                'expires_at': new_token_info.expires_at
            }) \
            .match({
                'provider': 'microsoft',
                'user_id': token_info.user_id
            }) \
            .execute()
            
        logger.info("Successfully updated Microsoft token in database")
        return new_token_info
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to refresh Microsoft token - HTTP error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error refreshing Microsoft token: {str(e)}")
        return None

async def get_google_title(url: str, token_info: TokenInfo, supabase: SupabaseClient) -> OnlineSheet | None:
    """Get document title using Google Drive API"""
    logger.info(f"Fetching Google doc title for URL: {url}")
    try:
        # Check if token is expired
        expires_at = datetime.fromisoformat(token_info.expires_at.replace('Z', '+00:00'))
        if expires_at <= datetime.now(timezone.utc):
            logger.warning("Google token is expired, attempting refresh")
            token_info = await refresh_google_token(token_info, supabase)
            if not token_info:
                logger.error("Failed to refresh Google token")
                return None

        # Create credentials object from token
        creds = Credentials(
            token=token_info.access_token,
            refresh_token=token_info.refresh_token,
            token_uri='https://oauth2.googleapis.com/token',
            client_id=os.getenv('GOOGLE_CLIENT_ID'),
            client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
            scopes=token_info.scope.split(' ')
        )
        
        # Extract file ID from URL
        file_id = None
        if '/document/d/' in url:
            file_id = url.split('/document/d/')[1].split('/')[0]
        elif '/spreadsheets/d/' in url:
            file_id = url.split('/spreadsheets/d/')[1].split('/')[0]
        
        if not file_id:
            return None

        # Build the service
        drive_service = build('drive', 'v3', credentials=creds)
        sheet_id = url.split('/d/')[1].split('/')[0]
        sheet_service = build('sheets', 'v4', credentials=creds)


        
        try:
            # Get file metadata
            file = drive_service.files().get(fileId=file_id, fields='name').execute()
            # Get sheet name from URL or fetch first sheet if not specified
            sheet_name = None
            if '#gid=' in url:
                gid = url.split('#gid=')[1]
                # Get spreadsheet metadata to find sheet name
                sheet_metadata = sheet_service.spreadsheets().get(spreadsheetId=sheet_id).execute()
                for sheet in sheet_metadata.get('sheets', ''):
                    if sheet.get('properties', {}).get('sheetId') == int(gid):
                        sheet_name = sheet['properties']['title']
                        break
            
            if not sheet_name:
                # If no specific sheet found, get the first sheet name
                sheet_metadata = sheet_service.spreadsheets().get(spreadsheetId=sheet_id).execute()
                sheet_name = sheet_metadata['sheets'][0]['properties']['title']
            doc_name = file.get('name')
            sheet_names = []
            sheet_names.append(sheet_name)
            sheet_md = OnlineSheet(doc_name=doc_name, provider='google', sheet_names=sheet_names) 

            return sheet_md

        except Exception as api_error:
            # Handle specific Google API errors
            if 'invalid_grant' in str(api_error):
                logger.warning("Invalid grant error, attempting token refresh")
                token_info = await refresh_google_token(token_info, supabase)
                if token_info:
                    # Retry once with new token
                    creds = Credentials(
                        token=token_info.access_token,
                        refresh_token=token_info.refresh_token,
                        token_uri='https://oauth2.googleapis.com/token',
                        client_id=os.getenv('GOOGLE_CLIENT_ID'),
                        client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
                        scopes=token_info.scope.split(' ')
                    )
                    drive_service = build('drive', 'v3', credentials=creds)
                    file = drive_service.files().get(fileId=file_id, fields='name').execute()
                    return file.get('name')
            raise  # Re-raise if it's not an invalid_grant error or if retry failed

    except Exception as e:
        logger.error(f"Error fetching Google doc title: {str(e)}")
        return None  # Return None instead of the error string to maintain consistency

async def get_microsoft_title(url: str, token_info: TokenInfo, supabase: SupabaseClient) -> OnlineSheet| None:
    """Get document title using Microsoft Graph API"""
    try:
        # Check if token is expired
        expires_at = datetime.fromisoformat(token_info.expires_at.replace('Z', '+00:00'))
        if expires_at <= datetime.now(timezone.utc):
            logger.warning("Microsoft token is expired, attempting refresh")
            token_info = await refresh_microsoft_token(token_info, supabase)
            if not token_info:
                logger.error("Failed to refresh Microsoft token")
                return None

        # Extract item ID from URL
        item_id = None
        if 'id=' in url:
            # Handle OneDrive URLs
            item_id = url.split('id=')[1].split('&')[0]
        elif '://' in url:
            # Handle SharePoint URLs
            path_parts = url.split('://')[-1].split('/')
            for i, part in enumerate(path_parts):
                if part in ['view.aspx', 'edit.aspx']:
                    item_id = path_parts[i-1]
                    break

        logger.info(f"Extracted item ID: {item_id}")
        if not item_id:
            logger.error(f"Could not extract item ID from URL: {url}")
            return None

        # First attempt with current token
        headers = {
            'Authorization': f"{token_info.token_type} {token_info.access_token}",
            'Content-Type': 'application/json'
        }
        
        api_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}"
        response = requests.get(api_url, headers=headers)
        
        # If unauthorized, try refreshing token and retry
        if response.status_code == 401:
            logger.warning("Received 401 error, attempting token refresh")
            token_info = await refresh_microsoft_token(token_info, supabase)
            if not token_info:
                logger.error("Failed to refresh Microsoft token after 401")
                return None
                
            # Retry with new token
            headers['Authorization'] = f"{token_info.token_type} {token_info.access_token}"
            response = requests.get(api_url, headers=headers)
        
        response.raise_for_status()
        file_data = response.json()
        file_name = file_data.get('name')

        # If it's an Excel file, get the active sheet name
        if file_data.get('file', {}).get('mimeType') == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            workbook_api_url = f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}/workbook/worksheets"
            workbook_response = requests.get(workbook_api_url, headers=headers)
            workbook_response.raise_for_status()
            sheets_data = workbook_response.json()
            
        sheet_names = []
        for sheet in sheets_data.get('value', []):
            sheet_names.append(sheet['name'])
        if "." in file_name:
            file_name = file_name.split(".")[0]
        
        sheet_md = OnlineSheet(doc_name=file_name, provider='microsoft', sheet_names=sheet_names) 
        return sheet_md

    except Exception as e:
        logger.error(f"Error fetching Microsoft doc title: {str(e)}")
        return None

@router.post("/get_document_title", response_model=WorkbookResponse)
async def get_document_title(
    url: DocumentTitleRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)]
):
    logger.info(f"Processing document title request for user {user_id}")
    if not user_id:
        return WorkbookResponse(
            url=url.url,
            success=False,
            error="Authentication required"
        )

    # Handle Google URLs
    if any(domain in url.url for domain in ['docs.google.com', 'sheets.google.com']):
        google_token = await get_provider_token(user_id, 'google', supabase)
        if not google_token:
            return WorkbookResponse(
                url=url.url,
                success=False,
                error="Google authentication required. Please connect your Google account."
            )

        try:
            online_sheet = await get_google_title( url.url, google_token, supabase)
            if online_sheet is None:
                return WorkbookResponse(
                    url=url.url,
                    success=False,
                    error="Error accessing Google Sheets. Please reconnect your Google account."
                )
            logger.info(f"Retrieved title: {online_sheet.doc_name} sheets: {online_sheet.sheet_names}")

            return WorkbookResponse(
                url=url.url,
                doc_name=online_sheet.doc_name,
                provider=online_sheet.provider,
                sheet_names=online_sheet.sheet_names,
                success=True
            )
        except Exception as e:
            logger.error(f"Error processing Google URL: {str(e)}")
            return WorkbookResponse(
                url=url.url,
                success=False,
                error="Error accessing Google Sheets. Please reconnect your Google account."
            )
        
    # Handle Microsoft URLs
    elif any(domain in url.url for domain in ['office.com', 'live.com', 'onedrive.live.com']):
        microsoft_token = await get_provider_token(user_id, 'microsoft', supabase)
        logger.info(f"Retrieved Microsoft token")
        if not microsoft_token:
            return WorkbookResponse(
                url=url.url,
                success=False,
                error="Microsoft authentication required. Please connect your Microsoft account."
            )

        online_sheet = await get_microsoft_title(url.url, microsoft_token, supabase)
        if online_sheet is not None:
            logger.info(f"Retrieved title: {online_sheet.doc_name} sheets: {online_sheet.sheet_names}")
            return WorkbookResponse(
                url=url.url,
                doc_name=online_sheet.doc_name,
                provider=online_sheet.provider,
                sheet_names=online_sheet.sheet_names,
                success=True
            )
        else:
            logger.error(f"Failed to retrieve title for Microsoft document: {url.url}")
            return WorkbookResponse(
                url=url.url,
                success=False,
                error="Error accessing Excel Online. Please reconnect your Microsoft account."
            )
    else:
        return WorkbookResponse(
            url=url.url,
            success=False,
            error="Unsupported document type"
        )