from fastapi import APIRouter, HTTPException, Depends, Security, Header
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
from app.utils.auth import get_current_user, get_supabase_client, security
import logging
from googleapiclient.errors import HttpError
import asyncio
from fastapi.security import HTTPAuthorizationCredentials
import aiohttp

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
    try:
        # Extract file ID from URL
        file_id = None
        if '/document/d/' in url:
            file_id = url.split('/document/d/')[1].split('/')[0]
        elif '/spreadsheets/d/' in url:
            file_id = url.split('/spreadsheets/d/')[1].split('/')[0]
        
        if not file_id:
            logger.error(f"Could not extract file ID from URL: {url}")
            return None

        # For direct access tokens from Google Picker, make direct API calls
        if not token_info.refresh_token:
            logger.info(f"Using direct access token (first 10 chars): {token_info.access_token[:10]}...")
            logger.info(f"Token scope: {token_info.scope}")
            
            # Add a small delay to allow for permission propagation
            await asyncio.sleep(2)
            logger.info("Waited 2 seconds for permission propagation")
            
            # Verify token is valid
            verify_url = 'https://www.googleapis.com/oauth2/v1/tokeninfo'
            params = {'access_token': token_info.access_token}
            async with aiohttp.ClientSession() as session:
                async with session.get(verify_url, params=params) as response:
                    logger.info(f"Token verification response status: {response.status}")
                    verify_response_text = await response.text()
                    logger.info(f"Token verification response: {verify_response_text}")
                    
                headers = {
                    'Authorization': f'Bearer {token_info.access_token}',
                    'Accept': 'application/json',
                }
                logger.info(f"Request headers: {headers}")
                
                # Try sheets API first with different format
                sheets_url = f'https://sheets.googleapis.com/v4/spreadsheets/{file_id}'  # Removed fields parameter
                logger.info(f"Making sheets request to: {sheets_url}")
                async with session.get(sheets_url, headers=headers) as response:
                    logger.info(f"Sheets response status: {response.status}")
                    sheets_response_text = await response.text()
                    
                    if response.status == 404:
                        error_data = await response.json()
                        error_message = error_data.get('error', {}).get('message', 'Unknown error')
                        error_reason = error_data.get('error', {}).get('errors', [{}])[0].get('reason', 'unknown')
                        logger.error(f"File not found: {file_id}. Reason: {error_reason}. Message: {error_message}")
                        
                        # Fall back to drive API
                        metadata_url = f'https://www.googleapis.com/drive/v3/files/{file_id}?fields=name,trashed,capabilities&supportsAllDrives=true'
                        logger.info(f"Falling back to Drive API: {metadata_url}")
                        async with session.get(metadata_url, headers=headers) as response:
                            logger.info(f"Metadata response status: {response.status}")
                            response_text = await response.text()
                            logger.info(f"Metadata response text: {response_text}")
                            
                            if response.status == 404:
                                error_data = await response.json()
                                error_message = error_data.get('error', {}).get('message', 'Unknown error')
                                error_reason = error_data.get('error', {}).get('errors', [{}])[0].get('reason', 'unknown')
                                logger.error(f"File not found: {file_id}. Reason: {error_reason}. Message: {error_message}")
                                return "not_found"
                            if response.status == 401:
                                logger.error(f"401 Unauthorized response: {response_text}")
                                raise HTTPException(status_code=401, detail="Invalid Google access token")
                            response.raise_for_status()
                            file_data = await response.json()
                            logger.info(f"File metadata response: {file_data}")
                            doc_name = file_data.get('name')

                        result = OnlineSheet(doc_name=doc_name, provider='google', sheet_names=[])
                        logger.info(f"Returning result: {result}")
                        return result
                    if response.status == 401:
                        logger.error(f"401 Unauthorized response: {sheets_response_text}")
                        raise HTTPException(status_code=401, detail="Invalid Google access token")
                    response.raise_for_status()
                    sheets_data = await response.json()
                    sheet_names = [sheet['properties']['title'] for sheet in sheets_data.get('sheets', [])]
                    logger.info(f"Extracted sheet names: {sheet_names}")

            result = OnlineSheet(doc_name=sheets_data.get('properties', {}).get('title'), provider='google', sheet_names=sheet_names)
            logger.info(f"Returning result: {result}")
            return result

        # For stored tokens with refresh capability, use the existing flow
        else:
            expires_at = datetime.fromisoformat(token_info.expires_at.replace('Z', '+00:00'))
            if expires_at <= datetime.now(timezone.utc):
                logger.warning("Google token is expired, attempting refresh")
                token_info = await refresh_google_token(token_info, supabase)
                if not token_info:
                    logger.error("Failed to refresh Google token")
                    return None

            # Create credentials object with refresh capability
            creds = Credentials(
                token=token_info.access_token,
                refresh_token=token_info.refresh_token,
                token_uri='https://oauth2.googleapis.com/token',
                client_id=os.getenv('GOOGLE_CLIENT_ID'),
                client_secret=os.getenv('GOOGLE_CLIENT_SECRET'),
                scopes=token_info.scope.split(' ')
            )
            
            # Build the service
            drive_service = build('drive', 'v3', credentials=creds)
            sheet_service = build('sheets', 'v4', credentials=creds)

            try:
                # Get file metadata
                file = drive_service.files().get(
                    fileId=file_id, 
                    fields='name',
                    supportsAllDrives=True
                ).execute()
                
                # Get sheet names
                sheet_metadata = sheet_service.spreadsheets().get(spreadsheetId=file_id).execute()
                sheet_names = [sheet['properties']['title'] for sheet in sheet_metadata.get('sheets', [])]

                return OnlineSheet(doc_name=file.get('name'), provider='google', sheet_names=sheet_names)

            except Exception as api_error:
                logger.error(f"API error while accessing file {file_id}: {str(api_error)}")
                if isinstance(api_error, HttpError) and api_error.resp.status == 404:
                    return "not_found"
                raise

    except Exception as e:
        logger.error(f"Error fetching Google doc title: {str(e)}")
        if str(e).startswith("<HttpError 404"):
            return "not_found"
        return None

async def get_microsoft_title(url: str, token_info: TokenInfo, supabase: SupabaseClient) -> OnlineSheet:
    """Get document title using Microsoft Graph API"""
    # Check if token is expired
    expires_at = datetime.fromisoformat(token_info.expires_at.replace('Z', '+00:00'))
    if expires_at <= datetime.now(timezone.utc):
        logger.warning("Microsoft token is expired, attempting refresh")
        token_info = await refresh_microsoft_token(token_info, supabase)
        if not token_info:
            raise Exception("Failed to refresh Microsoft token")

    # Extract item ID from URL
    item_id = None
    logger.info(f"Attempting to extract item ID from URL: {url}")
    
    if 'resid=' in url:
        # Handle Office 365/OneDrive URLs
        resid = url.split('resid=')[1].split('&')[0]
        # For OneDrive, we need to encode the resid differently
        if 'onedrive.live.com' in url:
            # Extract the personal identifier if present
            personal_id = None
            if '/personal/' in url:
                personal_id = url.split('/personal/')[1].split('/')[0]
                logger.info(f"Extracted personal ID: {personal_id}")
            
            # Use the driveId/items/resid format for OneDrive
            if personal_id:
                item_id = resid
                # Try the /me/drive endpoint first
                graph_endpoint = os.getenv('MS_GRAPH_ENDPOINT', 'https://graph.microsoft.com')
                api_url = f"{graph_endpoint}/v1.0/me/drive"
            else:
                item_id = resid
        else:
            item_id = resid
    elif 'id=' in url:
        # Handle other OneDrive URLs
        item_id = url.split('id=')[1].split('&')[0]
    elif '://' in url:
        # Handle SharePoint URLs
        path_parts = url.split('://')[-1].split('/')
        for i, part in enumerate(path_parts):
            if part.lower() in ['view.aspx', 'edit.aspx']:
                item_id = path_parts[i-1]
                break
            # Look for a GUID pattern
            elif len(part) > 30 and ('-' in part or '.' in part):
                item_id = part
                break

    if not item_id:
        raise ValueError(f"Could not extract item ID from URL: {url}")
    
    logger.info(f"Extracted item ID: {item_id}")

    # First attempt with current token
    headers = {
        'Authorization': f"{token_info.token_type} {token_info.access_token}",
        'Content-Type': 'application/json'
    }
    
    graph_endpoint = os.getenv('MS_GRAPH_ENDPOINT', 'https://graph.microsoft.com')
    
    # Log token info (safely)
    logger.info(f"Token type: {token_info.token_type}")
    logger.info(f"Token scopes: {token_info.scope}")
    logger.info(f"Environment variables: MS_GRAPH_ENDPOINT={graph_endpoint}")
    
    # For OneDrive personal, try to get the drive info first
    if 'onedrive.live.com' in url:
        logger.info("OneDrive personal URL detected, getting drive info first")
        drive_url = f"{graph_endpoint}/v1.0/me/drive"
        logger.info(f"Request headers for drive info: {headers}")
        drive_response = requests.get(drive_url, headers=headers)
        logger.info(f"Drive info response status: {drive_response.status_code}")
        logger.info(f"Drive info response headers: {dict(drive_response.headers)}")
        logger.info(f"Drive info response: {drive_response.text}")
        
        if drive_response.status_code == 200:
            # For personal OneDrive, always use /me/drive/items format
            item_id = item_id.lower()  # Ensure consistent casing
            api_url = f"{graph_endpoint}/v1.0/me/drive/items/{item_id}"
            logger.info(f"Using personal OneDrive URL: {api_url}")
        else:
            # Fallback to /me/drive/items format
            api_url = f"{graph_endpoint}/v1.0/me/drive/items/{item_id}"
    else:
        api_url = f"{graph_endpoint}/v1.0/me/drive/items/{item_id}"
    
    logger.info(f"Making Microsoft Graph API request to: {api_url}")
    logger.info(f"Using item_id: {item_id}")
    response = requests.get(api_url, headers=headers)
    logger.info(f"Microsoft Graph API response status: {response.status_code}")
    logger.info(f"Microsoft Graph API response: {response.text}")
    
    # If unauthorized, try refreshing token and retry
    if response.status_code == 401:
        logger.warning("Received 401 error, attempting token refresh")
        token_info = await refresh_microsoft_token(token_info, supabase)
        if not token_info:
            raise Exception("Failed to refresh Microsoft token after 401")
            
        # Retry with new token
        headers['Authorization'] = f"{token_info.token_type} {token_info.access_token}"
        response = requests.get(api_url, headers=headers)
    
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if response.status_code == 404:
            raise Exception("Document not found or not accessible")
        raise Exception(f"HTTP error occurred: {str(e)}")

    file_data = response.json()
    file_name = file_data.get('name')

    # If it's an Excel file, get the active sheet name
    sheet_names = []
    if file_data.get('file', {}).get('mimeType') == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
        workbook_api_url = f"{graph_endpoint}/v1.0/me/drive/items/{item_id}/workbook/worksheets"
        max_retries = 3
        retry_count = 0
        retry_delay = 1  # Start with 1 second delay
        
        while retry_count < max_retries:
            try:
                logger.info(f"Attempting to fetch worksheets (attempt {retry_count + 1}/{max_retries})")
                workbook_response = requests.get(workbook_api_url, headers=headers)
                
                # If we get a 503, wait and retry
                if workbook_response.status_code == 503:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Received 503 error, retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                
                workbook_response.raise_for_status()
                sheets_data = workbook_response.json()
                for sheet in sheets_data.get('value', []):
                    sheet_names.append(sheet['name'])
                break  # Success, exit the retry loop
                
            except requests.exceptions.HTTPError as e:
                if workbook_response.status_code == 503 and retry_count < max_retries - 1:
                    continue  # Try again if we have retries left
                logger.error(f"Failed to fetch worksheet information after {retry_count + 1} attempts: {str(e)}")
                # Don't raise an exception, just log the error and continue with empty sheet names
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching worksheets: {str(e)}")
                break
        
        if not sheet_names:
            logger.warning("Could not fetch worksheet names, continuing with file name only")
    
    if "." in file_name:
        file_name = file_name.split(".")[0]
    
    sheet_md = OnlineSheet(doc_name=file_name, provider='microsoft', sheet_names=sheet_names) 
    return sheet_md

@router.post("/get_sheet_names", response_model=WorkbookResponse)
async def get_sheet_names(
    url: DocumentTitleRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)],
    access_token: str | None = Header(None, alias="Access-Token"),
    provider: str | None = Header(None)
):
    if not user_id:
        return WorkbookResponse(
            url=url.url,
            success=False,
            error="Authentication required"
        )

    if not provider:
        return WorkbookResponse(
            url=url.url,
            success=False,
            error="Provider is required"
        )

    provider = provider.lower()
    
    # First try to get token from database
    token_info = await get_provider_token(user_id, provider, supabase)
    
    # If no database token and no access token provided, return error
    if not token_info and not access_token:
        return WorkbookResponse(
            url=url.url,
            success=False,
            error=f"No {provider} access token found. Please reconnect your {provider} account."
        )
    
    # If no database token but access token provided, create temporary token
    if not token_info and access_token:
        token_info = TokenInfo(
            access_token=access_token,
            refresh_token="",  # Not needed for temporary access
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            token_type="Bearer",
            scope="Files.Read.Selected" if provider == "microsoft" else "https://www.googleapis.com/auth/drive.file",
            user_id=user_id
        )

    try:
        if provider == "google":
            logger.info(f"Attempting to get Google sheet names for URL: {url.url}")
            online_sheet = await get_google_title(url.url, token_info, supabase)
            if online_sheet == "not_found":
                error_message = "Document not found or not accessible. Please check that the document exists and you have been granted access to it."
                logger.error(f"Document not found or not accessible: {url.url}")
                return WorkbookResponse(
                    url=url.url,
                    success=False,
                    error=error_message
                )
            if online_sheet is None:
                logger.error(f"Error accessing Google Sheets: {url.url}")
                return WorkbookResponse(
                    url=url.url,
                    success=False,
                    error="Error accessing Google Sheets. Please check your permissions and try again."
                )

            logger.info(f"Successfully retrieved sheet info: {online_sheet}")
            return WorkbookResponse(
                url=url.url,
                doc_name=online_sheet.doc_name,
                provider=online_sheet.provider,
                sheet_names=online_sheet.sheet_names,
                success=True
            )
        elif provider == "microsoft":
            online_sheet = await get_microsoft_title(url.url, token_info, supabase)
            logger.info(f"Retrieved title: {online_sheet.doc_name} sheets: {online_sheet.sheet_names}")
            return WorkbookResponse(
                url=url.url,
                doc_name=online_sheet.doc_name,
                provider=online_sheet.provider,
                sheet_names=online_sheet.sheet_names,
                success=True
            )
        else:
            return WorkbookResponse(
                url=url.url,
                success=False,
                error=f"Unsupported provider: {provider}"
            )
    except Exception as e:
        logger.error(f"Error processing {provider} URL: {str(e)}")
        return WorkbookResponse(
            url=url.url,
            success=False,
            error=f"Error accessing {provider} file. Please try selecting the file again."
        )