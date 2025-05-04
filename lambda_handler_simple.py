import json
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create a minimal FastAPI app for testing - keep it very simple
app = FastAPI(
    title="SheetAssist API",
    description="SheetAssist API Lambda Version"
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
def root():
    logger.info("Root endpoint called")
    return {"message": "SheetAssist API running on Lambda!"}

@app.get("/health")
def health():
    logger.info("Health endpoint called")
    return {"status": "healthy"}

# Wrap the app with Mangum for Lambda compatibility
handler = Mangum(app, lifespan="off") 