from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase.client import Client as SupabaseClient
from typing import Annotated
import jwt
import os
from supabase import create_client, Client


security = HTTPBearer()



async def get_supabase_client() -> SupabaseClient:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return supabase

async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(security)],
    supabase: Annotated[SupabaseClient, Depends(get_supabase_client)]
) -> str:
    """
    Validate JWT token from Supabase and return user_id
    """
    try:
        # Get token from Authorization header
        token = credentials.credentials
        SUPABASE_JWT_SECRET = os.environ.get("SUPABASE_JWT_SECRET")

        # Verify token using Supabase JWT secret
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,  # Your Supabase JWT secret
            algorithms=["HS256"],
            audience="authenticated"
        )
        
        user_id = payload.get('sub')
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )

        # Verify user exists in Supabase
        response = supabase.from_('user_profile').select('id').eq('id', user_id).single()
        if not response.data:
            raise HTTPException(
                status_code=401,
                detail="User not found"
            )

        return user_id

    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=str(e)
        )