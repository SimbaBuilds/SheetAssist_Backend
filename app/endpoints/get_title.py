from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from msal import ConfidentialClientApplication
import json
import requests
from datetime import datetime
from typing import Annotated
from supabase.client import Client as SupabaseClient
from app.utils.auth import get_current_user, get_supabase_client

router = APIRouter()

class URLRequest(BaseModel):
    urls: List[str]

class DocumentTitle(BaseModel):
    url: str
    title: str

class TokenData(BaseModel):
    google_token: dict | None = None
    microsoft_token: dict | None = None

async def get_tokens(user_id: str, supabase_client) -> TokenData:
    """Fetch tokens from your token storage (Supabase)"""
    try:
        response = await supabase_client.from_('user_tokens').select('*').eq('user_id', user_id).single()
        if response.data:
            return TokenData(
                google_token=json.loads(response.data.get('google_token', '{}')),
                microsoft_token=json.loads(response.data.get('microsoft_token', '{}'))
            )
        return TokenData()
    except Exception as e:
        raise HTTPException(status_code=401, detail="Could not fetch tokens")

async def get_google_title(url: str, credentials: dict) -> str | None:
    """Get document title using Google Drive API"""
    try:
        # Create credentials object from token
        creds = Credentials.from_authorized_user_info(credentials)
        
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

async def get_microsoft_title(url: str, token: dict) -> str | None:
    """Get document title using Microsoft Graph API"""
    try:
        # Check if token is expired and needs refresh
        if datetime.fromtimestamp(token.get('expires_at', 0)) <= datetime.now():
            # Implement token refresh logic here
            pass

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
            'Authorization': f"Bearer {token['access_token']}",
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
    # Your existing code here
    # user_id will be automatically extracted from the JWT
    tokens = await get_tokens(user_id, supabase)
    titles = []
    
    for url in request.urls:
        title = None
        
        # Handle Google URLs
        if any(domain in url for domain in ['docs.google.com', 'sheets.google.com']):
            if not tokens.google_token:
                raise HTTPException(
                    status_code=401,
                    detail="Google authentication required"
                )
            title = await get_google_title(url, tokens.google_token)
            
        # Handle Microsoft URLs
        elif any(domain in url for domain in ['office.com', 'live.com', 'sharepoint.com']):
            if not tokens.microsoft_token:
                raise HTTPException(
                    status_code=401,
                    detail="Microsoft authentication required"
                )
            title = await get_microsoft_title(url, tokens.microsoft_token)
            
        titles.append(DocumentTitle(
            url=url,
            title=title or url  # Fallback to URL if title fetch fails
        ))
    
    return titles