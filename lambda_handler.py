import sys
import os
import logging
from dotenv import load_dotenv
import traceback
import json

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
base_handler = Mangum(app, lifespan="auto")

# List of allowed origins
ALLOWED_ORIGINS = [
    "https://localhost:3000",
    "http://localhost:3000",
    "https://aidocassist.com",
    "https://sheetassistapp.com",
    "https://www.aidocassist.com",
    "https://api.aidocassist.com",
    "http://api.aidocassist.com",
    "https://1t2fkfa15h.execute-api.us-east-1.amazonaws.com"
]

# Custom handler with CORS support
def handler(event, context):
    # Print event for debugging
    logger.info(f"Event received: {json.dumps(event)}")
    
    # Get the origin from the request headers
    request_origin = event.get('headers', {}).get('origin') or event.get('headers', {}).get('Origin')
    logger.info(f"Request origin: {request_origin}")
    
    # Check if this is an OPTIONS preflight request
    is_options = event.get('requestContext', {}).get('http', {}).get('method') == 'OPTIONS'
    
    # Check if the origin is allowed
    allow_origin = request_origin if request_origin in ALLOWED_ORIGINS else ALLOWED_ORIGINS[0]
    
    # Call the base handler
    response = base_handler(event, context)
    
    # Always include CORS headers in the response
    headers = response.get('headers', {})
    
    # Ensure CORS headers exist
    cors_headers = {
        'Access-Control-Allow-Origin': allow_origin,
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token,X-Amz-User-Agent',
        'Access-Control-Allow-Methods': 'GET,POST,PUT,DELETE,OPTIONS',
        'Access-Control-Allow-Credentials': 'true',
        'Access-Control-Max-Age': '86400',
    }
    
    # Update the response headers
    headers.update(cors_headers)
    response['headers'] = headers
    
    # Special handling for OPTIONS requests
    if is_options:
        logger.info("Received OPTIONS request, returning CORS headers")
        return {
            'statusCode': 200,
            'headers': cors_headers,
            'body': ''
        }
    
    logger.info(f"Returning response with CORS headers: {response}")
    return response

# For local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 