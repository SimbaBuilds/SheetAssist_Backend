from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
import requests
import os
from datetime import datetime, timezone
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
        logger.info(f"Successfully retrieved {provider} token")
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
        
        # Create new token info
        new_token_info = TokenInfo(
            access_token=creds.token,
            refresh_token=creds.refresh_token or token_info.refresh_token,
            token_type='Bearer',
            scope=' '.join(creds.scopes),
            expires_at=(datetime.now(timezone.utc) + 
                       datetime.timedelta(seconds=creds.expiry.timestamp() - datetime.now().timestamp())
                      ).isoformat()
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
            
        logger.info("Successfully refreshed Google token")
        return new_token_info
        
    except RefreshError as e:
        logger.error(f"Failed to refresh token - token may be revoked: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error refreshing Google token: {str(e)}")
        return None

async def get_google_title(url: str, token_info: TokenInfo, supabase: SupabaseClient) -> str | None:
    """Get document title using Google Drive API"""
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
        
        # Get file metadata
        file = service.files().get(fileId=file_id, fields='name').execute()
        return file.get('name')

    except Exception as e:
        print(f"Error fetching Google doc title: {e}")
        return None

async def get_microsoft_title(url: str, token_info: TokenInfo) -> str | None:
    """Get document title using Microsoft Graph API"""
    try:
        # Check if token is expired
        expires_at = datetime.fromisoformat(token_info.expires_at.replace('Z', '+00:00'))
        if expires_at <= datetime.now(timezone.utc):
            # Token is expired, should implement refresh logic here
            print("Microsoft token is expired")
            return None

        # Extract item ID from URL
        item_id = None
        if '://' in url:
            path_parts = url.split('://')[-1].split('/')
            for i, part in enumerate(path_parts):
                if part in ['view.aspx', 'edit.aspx']:
                    item_id = path_parts[i-1]
                    break

        if not item_id:
            return None

        # Call Microsoft Graph API
        headers = {
            'Authorization': f"{token_info.token_type} {token_info.access_token}",
            'Content-Type': 'application/json'
        }
        
        response = requests.get(
            f"https://graph.microsoft.com/v1.0/me/drive/items/{item_id}",
            headers=headers
        )
        response.raise_for_status()
        
        return response.json().get('name')

    except Exception as e:
        print(f"Error fetching Microsoft doc title: {e}")
        return None

@router.post("/get_document_titles", response_model=List[DocumentTitle])
async def get_document_titles(
    request: URLRequest,
    user_id: Annotated[str, Depends(get_current_user)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)]
):
    logger.info(f"Processing document titles request for user {user_id}")
    if not user_id:
        logger.error("No user_id provided")
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )

    titles = []
    
    for url in request.urls:
        title = None
        
        # Handle Google URLs
        if any(domain in url for domain in ['docs.google.com', 'sheets.google.com']):
            google_token = await get_provider_token(user_id, 'google', supabase)
            if not google_token:
                logger.error(f"Google token not found for user {user_id}")
                raise HTTPException(
                    status_code=401,
                    detail="Google authentication required. Please connect your Google account."
                )
            title = await get_google_title(url, google_token, supabase)
            
        # Handle Microsoft URLs
        elif any(domain in url for domain in ['office.com', 'live.com', 'onedrive.live.com']):
            microsoft_token = await get_provider_token(user_id, 'microsoft', supabase)
            if not microsoft_token:
                logger.error(f"Microsoft token not found for user {user_id}")
                raise HTTPException(
                    status_code=401,
                    detail="Microsoft authentication required. Please connect your Microsoft account."
                )
            title = await get_microsoft_title(url, microsoft_token)
            
        titles.append(DocumentTitle(
            url=url,
            title=title or url  # Fallback to URL if title fetch fails
        ))
    
    return titles