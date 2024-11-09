from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
import os
from typing import Dict

def refresh_google_access_token(refresh_token: str) -> Dict:
    """Refresh Google access token using refresh token."""
    
    # Create credentials object
    creds = Credentials(
        None,  # No access token since it's expired
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv('GOOGLE_CLIENT_ID'),
        client_secret=os.getenv('GOOGLE_CLIENT_SECRET')
    )
    
    # Refresh the credentials
    creds.refresh(Request())
    
    # Return new tokens
    return {
        'access_token': creds.token,
        'refresh_token': refresh_token,  # Same refresh token
        'expiry_date': int(creds.expiry.timestamp())
    }

def create_google_oauth_flow() -> Flow:
    """Create Google OAuth flow for initial authorization."""
    
    return Flow.from_client_config(
        {
            "web": {
                "client_id": os.getenv('GOOGLE_CLIENT_ID'),
                "client_secret": os.getenv('GOOGLE_CLIENT_SECRET'),
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        },
        scopes=[
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/documents',
            'https://www.googleapis.com/auth/drive'
        ],
        redirect_uri=os.getenv('GOOGLE_REDIRECT_URI')
    )