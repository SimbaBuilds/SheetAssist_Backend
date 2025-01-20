import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.endpoints import process_query, download, get_doc_title, data_visualization
from app.utils.file_management import temp_file_manager
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await temp_file_manager.start_periodic_cleanup()
    yield
    # Shutdown
    await temp_file_manager.stop_periodic_cleanup()

app = FastAPI(lifespan=lifespan)

# Configure CORS with specific origins
origins = [
    "https://aidocassist.com",
    "https://www.aidocassist.com",
    "https://api.aidocassist.com",
    "https://localhost:3000",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

@app.get("/")
async def root():
    return {"message": "SS Assist API"}

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