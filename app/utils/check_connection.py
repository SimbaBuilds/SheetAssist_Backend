from fastapi import Request, HTTPException
import logging

logger = logging.getLogger(__name__)

# Add client disconnection check
async def check_client_connection(request: Request):
    if await request.is_disconnected():
        logger.info(f"Client disconnected")
        raise HTTPException(status_code=499, detail="Client Closed Request")
