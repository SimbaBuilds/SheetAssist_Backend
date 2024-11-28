from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase.client import Client as SupabaseClient
from typing import Annotated
import jwt
import os
from supabase import create_client, Client
import logging

logger = logging.getLogger(__name__)

security = HTTPBearer()



async def get_supabase_client() -> SupabaseClient:
    SUPABASE_URL = os.environ.get("SUPABASE_URL")
    SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_ANON_KEY")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
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
        if not SUPABASE_JWT_SECRET:
            logger.error("SUPABASE_JWT_SECRET environment variable not found")
            raise HTTPException(
                status_code=401,
                detail="Server configuration error"
            )

        # Verify token using Supabase JWT secret
        try:
            payload = jwt.decode(
                token,
                SUPABASE_JWT_SECRET,
                algorithms=["HS256"],
                audience="authenticated"
            )
        except jwt.InvalidTokenError as e:
            logger.error(f"JWT decode failed: {str(e)}")
            raise
        
        user_id = payload.get('sub')
        if not user_id:
            logger.error("No user_id (sub) found in token payload")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )

        # Verify user exists in Supabase
        try:
            # Execute the query and get the response
            response = supabase.from_('user_profile').select('id').eq('id', user_id).execute()
            
            # Check if we got any data back
            if not response.data or len(response.data) == 0:
                logger.error(f"User {user_id} not found in user_profile table")
                raise HTTPException(
                    status_code=401,
                    detail="User not found"
                )
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            raise HTTPException(
                status_code=401,
                detail=f"Database error: {str(e)}"
            )

        logger.info("=== Authentication successful ===")
        return user_id

    except jwt.InvalidTokenError:
        logger.error("Invalid token error")
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    except Exception as e:
        logger.error(f"Unexpected authentication error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=str(e)
        )