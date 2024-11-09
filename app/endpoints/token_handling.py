from fastapi import APIRouter, HTTPException
from supabase import create_client, Client
from typing import Dict
import os
from datetime import datetime, timezone
from cryptography.fernet import Fernet
from app.schemas import TokenData
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

router = APIRouter()

# Initialize encryption key (store this securely!)
ENCRYPTION_KEY = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
fernet = Fernet(ENCRYPTION_KEY)

# Initialize Supabase client with service role
supabase: Client = create_client(
    os.getenv('NEXT_PUBLIC_SUPABASE_URL'),
    os.getenv('SUPABASE_SERVICE_ROLE_KEY')
)


@router.post("/store-google-tokens")
async def store_google_tokens(data: TokenData):
    try:
        # Encrypt tokens before storage
        encrypted_tokens = fernet.encrypt(str(data.tokens).encode())
        
        # Store encrypted tokens in Supabase
        response = supabase.table('google_tokens').upsert({
            'user_id': data.user_id,
            'encrypted_tokens': encrypted_tokens.decode(),
            'updated_at': datetime.now(timezone.utc).isoformat()
        }).execute()
        
        if hasattr(response, 'error') and response.error is not None:
            raise HTTPException(status_code=500, detail=str(response.error))
            
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/refresh-google-tokens/{user_id}")
async def refresh_google_tokens(user_id: str):
    try:
        # Fetch encrypted tokens
        response = supabase.table('google_tokens').select('encrypted_tokens').eq('user_id', user_id).execute()
        
        if not response.data:
            raise HTTPException(status_code=404, detail="Tokens not found")
            
        # Decrypt tokens
        encrypted_tokens = response.data[0]['encrypted_tokens']
        tokens_str = fernet.decrypt(encrypted_tokens.encode()).decode()
        tokens = eval(tokens_str)  # Convert string back to dict
        
        # Check if access token needs refresh
        if is_token_expired(tokens['expiry_date']):
            # Use refresh token to get new access token
            new_tokens = refresh_google_access_token(tokens['refresh_token'])
            
            # Store new tokens
            encrypted_new_tokens = fernet.encrypt(str(new_tokens).encode())
            supabase.table('google_tokens').upsert({
                'user_id': user_id,
                'encrypted_tokens': encrypted_new_tokens.decode(),
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).execute()
            
            return {"status": "success", "tokens": new_tokens}
            
        return {"status": "success", "tokens": tokens}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def is_token_expired(expiry_date: int) -> bool:
    return datetime.now(timezone.utc).timestamp() > expiry_date

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