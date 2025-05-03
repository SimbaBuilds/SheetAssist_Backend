import sys
import os
import logging
from dotenv import load_dotenv
import traceback

# Configure logging for Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Try to load environment variables from .env file if present (for local testing)
load_dotenv()

# Add the application directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from mangum import Mangum
    from app.main import app
    logger.info("Successfully imported application dependencies")
except Exception as e:
    logger.error(f"Error importing application dependencies: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# Create a Mangum adapter for the FastAPI app
handler = Mangum(app, lifespan="off")

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 