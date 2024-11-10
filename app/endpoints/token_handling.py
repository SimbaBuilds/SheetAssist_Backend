from fastapi import APIRouter, HTTPException
from supabase import create_client, Client
from typing import Dict
import os
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from app.schemas import TokenData
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from app.schemas import GoogleTokenRecord
from uuid import UUID

router = APIRouter()

# Initialize encryption key (store this securely!)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
fernet = Fernet(ENCRYPTION_KEY)

# Initialize Supabase client with service role
supabase: Client = create_client(
    os.getenv('NEXT_PUBLIC_SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)


@router.post("/auth/store-google-tokens")
async def store_google_tokens(data: TokenData):
    try:
        # Encrypt tokens separately
        encrypted_access_token = fernet.encrypt(data.tokens['access_token'].encode()).decode()
        encrypted_refresh_token = fernet.encrypt(data.tokens['refresh_token'].encode()).decode()
        
        # Create record matching GoogleTokenRecord schema
        token_record = {
            'user_id': UUID(data.user_id),
            'encrypted_access_token': encrypted_access_token,
            'encrypted_refresh_token': encrypted_refresh_token,
            'expiry_timestamp': datetime.fromtimestamp(data.tokens['expiry_date'], tz=timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
        
        # Store in Supabase
        response = supabase.table('google_tokens').upsert(token_record).execute()
        
        if hasattr(response, 'error') and response.error is not None:
            raise HTTPException(status_code=500, detail=str(response.error))
            
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/refresh-google-tokens/{user_id}")
async def refresh_google_tokens(user_id: str):
    try:
        # Fetch encrypted tokens
        response = supabase.table('google_tokens').select(
            'encrypted_access_token',
            'encrypted_refresh_token',
            'expiry_timestamp'
        ).eq('user_id', UUID(user_id)).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Tokens not found")
            
        # Decrypt tokens
        record = response.data[0]
        refresh_token = fernet.decrypt(record['encrypted_refresh_token'].encode()).decode()
        
        # Check if access token needs refresh
        if datetime.now(timezone.utc) > record['expiry_timestamp']:
            # Use refresh token to get new access token
            new_tokens = refresh_google_access_token(refresh_token)
            
            # Store new tokens
            token_record = {
                'user_id': UUID(user_id),
                'encrypted_access_token': fernet.encrypt(new_tokens['access_token'].encode()).decode(),
                'encrypted_refresh_token': fernet.encrypt(new_tokens['refresh_token'].encode()).decode(),
                'expiry_timestamp': datetime.fromtimestamp(new_tokens['expiry_date'], tz=timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            
            supabase.table('google_tokens').upsert(token_record).execute()
            
            return {"status": "success", "tokens": new_tokens}
        
        # If token is still valid, return current tokens
        access_token = fernet.decrypt(record['encrypted_access_token'].encode()).decode()
        current_tokens = {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'expiry_date': int(record['expiry_timestamp'].timestamp())
        }
        
        return {"status": "success", "tokens": current_tokens}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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