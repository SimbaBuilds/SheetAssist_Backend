import sys
import os
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
from app.endpoints import get_sheet_names, process_query, download, data_visualization
from app.utils.s3_file_management import temp_file_manager
from contextlib import asynccontextmanager

# # Configure logging - Uncomment if needed in Lambda
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler = logging.StreamHandler()
# console_handler.setFormatter(formatter)
# logging.basicConfig(level=logging.INFO, handlers=[console_handler])
# logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await temp_file_manager.start_periodic_cleanup()
    yield
    # Shutdown
    await temp_file_manager.stop_periodic_cleanup()

app = FastAPI(
    title="SheetAssist API",
    description="API for SheetAssist application",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

@app.middleware("http")
async def error_logging_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        # Log exception details for Lambda
        print(f"Request failed: {request.url}")
        print(f"Error details: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "traceback": traceback.format_exc()
            }
        )

# Configure CORS with specific origins
origins = [
    "https://aidocassist.com",
    "https://sheetassistapp.com",
    "https://www.aidocassist.com",
    "https://api.aidocassist.com",
    "http://api.aidocassist.com",
    "https://localhost:3000",
    "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
async def root():
    return {"message": "SheetAssist API - Lambda version"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(process_query.router)
app.include_router(download.router)
app.include_router(get_sheet_names.router)
app.include_router(data_visualization.router)

if __name__ == "__main__":    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)


#python app/main.py