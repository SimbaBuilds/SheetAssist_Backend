import sys
import os
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
# Only import uvicorn when running the script directly (not in Lambda)
import logging
from app.endpoints import get_sheet_names, process_query, download, data_visualization, health
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
        # Add CORS headers for HTTPS localhost
        if "origin" in request.headers and request.headers["origin"] == "https://localhost:3000":
            response.headers["Access-Control-Allow-Origin"] = "https://localhost:3000"
            response.headers["Access-Control-Allow-Credentials"] = "true"
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
            },
            headers={
                "Access-Control-Allow-Origin": "https://localhost:3000",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
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
    "http://localhost:3000",
    "http://localhost:8000",
    "https://1t2fkfa15h.execute-api.us-east-1.amazonaws.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=86400,
)

@app.options("/{path:path}")
async def options_route(path: str):
    """Handle OPTIONS requests for CORS preflight."""
    return PlainTextResponse(
        content="",
        headers={
            "Access-Control-Allow-Origin": "https://localhost:3000",  # Specifically for HTTPS localhost
            "Access-Control-Allow-Methods": "*",  # Allow all methods
            "Access-Control-Allow-Headers": "*",  # Allow all headers
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
        },
    )

@app.get("/")
async def root():
    return {"message": "SheetAssist API - Lambda version"}

app.include_router(process_query.router)
app.include_router(download.router)
app.include_router(get_sheet_names.router)
app.include_router(data_visualization.router)
app.include_router(health.router)

if __name__ == "__main__":    
    port = int(os.getenv("PORT", 8000))
    # Only import uvicorn when running locally
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)


#python app/main.py