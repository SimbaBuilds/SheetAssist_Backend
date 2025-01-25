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
from app.endpoints import process_query, download, get_doc_title, data_visualization
from app.utils.s3_file_management import temp_file_manager
from app.dev_utils.memory_profiler import MemoryProfilerMiddleware
from contextlib import asynccontextmanager

# Configure logging
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Configure console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(level=logging.INFO, handlers=[console_handler])
logger = logging.getLogger(__name__)

# Configure S3 logger
s3_logger = logging.getLogger("s3_operations")
s3_logger.setLevel(logging.INFO)
s3_logger.addHandler(console_handler)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup - initializing services")
    logger.info("Starting S3 temporary file management service")
    await temp_file_manager.start_periodic_cleanup()
    yield
    # Shutdown
    logger.info("Application shutdown initiated")
    logger.info("Stopping S3 temporary file management service")
    await temp_file_manager.stop_periodic_cleanup()
    logger.info("Application shutdown complete")

app = FastAPI(lifespan=lifespan)

# Add memory profiler middleware before other middleware
app.add_middleware(MemoryProfilerMiddleware)

@app.middleware("http")
async def error_logging_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as e:
        logger.error(f"Request failed: {request.url}")
        logger.error(f"Error details: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    return {"message": "FastAPI Config Test - Success!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

app.include_router(process_query.router)
app.include_router(download.router)
app.include_router(get_doc_title.router)
app.include_router(data_visualization.router)

if __name__ == "__main__":    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)


#python app/main.py