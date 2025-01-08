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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Specify domains if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(process_query.router)
app.include_router(download.router)
app.include_router(get_doc_title.router)
app.include_router(data_visualization.router)

if __name__ == "__main__":    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="localhost", port=port, reload=True)


#python app/main.py