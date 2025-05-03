import json
from fastapi import FastAPI
from mangum import Mangum

# Create a minimal FastAPI app for testing - keep it very simple
app = FastAPI(
    title="SheetAssist API",
    description="SheetAssist API Lambda Version"
)

@app.get("/")
def root():
    return {"message": "SheetAssist API running on Lambda!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

# Wrap the app with Mangum for Lambda compatibility
handler = Mangum(app, lifespan="off") 