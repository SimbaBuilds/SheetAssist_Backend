from fastapi import APIRouter, HTTPException

router = APIRouter()

@router.get("/health")
async def health_check():
    """A simple health check endpoint that doesn't use any complex dependencies"""
    try:
        return {"status": "healthy", "message": "API is running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 