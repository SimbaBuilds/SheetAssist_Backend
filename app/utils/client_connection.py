from fastapi import Request
import logging
import asyncio
from supabase import Client

logger = logging.getLogger(__name__)

# Add client disconnection check
async def check_client_connection(request: Request) -> bool:
    """Enhanced connection check with retry logic"""
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    for attempt in range(MAX_RETRIES):
        try:
            if await request.is_disconnected():
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY)
                    continue
                logger.warning("Client disconnected after retries")
                return False
            return True
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
                continue
            logger.error(f"Connection check error: {e}")
            return False
    
    return True



def check_job_status(job_id: str, supabase: Client) -> bool:
    # Check if job has been canceled
    job_status_check = supabase.table("jobs").select("status").eq("job_id", job_id).execute()
    if job_status_check.data[0]["status"] == "canceled":
        logger.info(f"Job {job_id} was canceled. Stopping processing.")
        return True
    return False


