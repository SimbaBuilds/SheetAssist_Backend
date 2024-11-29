from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
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

class URLRequest(BaseModel):
    urls: List[str]

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

class DocumentTitleResponse(BaseModel):
    url: str
    title: Optional[str] = None
    error: Optional[str] = None
    success: bool

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

async def get_google_title(url: str, token_info: TokenInfo, supabase: SupabaseClient) -> str | None:
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
        service = build('drive', 'v3', credentials=creds)
        
        try:
            # Get file metadata
            file = service.files().get(fileId=file_id, fields='name').execute()
            return file.get('name')
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
                    service = build('drive', 'v3', credentials=creds)
                    file = service.files().get(fileId=file_id, fields='name').execute()
                    return file.get('name')
            raise  # Re-raise if it's not an invalid_grant error or if retry failed

    except Exception as e:
        logger.error(f"Error fetching Google doc title: {str(e)}")
        return None  # Return None instead of the error string to maintain consistency

async def get_microsoft_title(url: str, token_info: TokenInfo, supabase: SupabaseClient) -> str | None:
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
        return response.json().get('name')

    except Exception as e:
        logger.error(f"Error fetching Microsoft doc title: {str(e)}")
        return None

@router.post("/get_document_titles", response_model=List[DocumentTitleResponse])
async def get_document_titles(
    request: URLRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)]
):
    logger.info(f"Processing document titles request for user {user_id}")
    if not user_id:
        return [DocumentTitleResponse(
            url=url,
            success=False,
            error="Authentication required"
        ) for url in request.urls]

    titles = []
    
    for url in request.urls:
        # Handle Google URLs
        if any(domain in url for domain in ['docs.google.com', 'sheets.google.com']):
            google_token = await get_provider_token(user_id, 'google', supabase)
            if not google_token:
                titles.append(DocumentTitleResponse(
                    url=url,
                    success=False,
                    error="Google authentication required. Please connect your Google account."
                ))
                continue

            try:
                title = await get_google_title(url, google_token, supabase)
                if title is None:
                    titles.append(DocumentTitleResponse(
                        url=url,
                        success=False,
                        error="Error accessing Google document. Please reconnect your Google account."
                    ))
                else:
                    titles.append(DocumentTitleResponse(
                        url=url,
                        title=title,
                        success=True
                    ))
            except Exception as e:
                logger.error(f"Error processing Google URL: {str(e)}")
                titles.append(DocumentTitleResponse(
                    url=url,
                    success=False,
                    error="Error accessing Google document. Please reconnect your Google account."
                ))
            
        # Handle Microsoft URLs
        elif any(domain in url for domain in ['office.com', 'live.com', 'onedrive.live.com']):
            microsoft_token = await get_provider_token(user_id, 'microsoft', supabase)
            logger.info(f"Retrieved Microsoft token: {microsoft_token}")
            if not microsoft_token:
                titles.append(DocumentTitleResponse(
                    url=url,
                    success=False,
                    error="Microsoft authentication required. Please connect your Microsoft account."
                ))
                continue

            title = await get_microsoft_title(url, microsoft_token, supabase)
            if title:
                logger.info(f"Retrieved title: {title}")
                titles.append(DocumentTitleResponse(
                    url=url,
                    title=title,
                    success=True
                ))
            else:
                logger.error(f"Failed to retrieve title for Microsoft document: {url}")
                titles.append(DocumentTitleResponse(
                    url=url,
                    success=False,
                    error="Error accessing Microsoft document. Please reconnect your Microsoft account."
                ))
        else:
            titles.append(DocumentTitleResponse(
                url=url,
                success=False,
                error="Unsupported document type"
            ))
    logger.info(f"Successfully processed document titles request with {len(titles)} titles\n {titles[0].url} and title {titles[0].title} and success {titles[0].success} and error {titles[0].error}")  
    return titles