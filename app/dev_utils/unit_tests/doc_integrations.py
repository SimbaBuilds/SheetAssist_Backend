# Replace the current sys.path modifications with this more robust approach
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

# Now we can import from app
from app.utils.google_integration import GoogleIntegration
from app.utils.microsoft_integration import MicrosoftIntegration
from app.schemas import FileDataInfo


import pytest
import pandas as pd
import os
import sys
from datetime import datetime, date
from dotenv import load_dotenv
import logging
from supabase import create_client


pytestmark = pytest.mark.asyncio


# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client directly
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

# Get test user ID from environment variable or set a test value
user_id = "695eadee-feda-492c-8f95-86f72fcc10c4"

# Test data
TEST_DF = pd.DataFrame({
    'Name': ['Test Receipt', 'Test Receipt 2'],
    'Age': [30, 25],
    'Date': [datetime.now(), datetime.now()]
})

TEST_DICT = {
    'name': 'Test Receipt',
    'age': 30,
    'date': datetime.now()
}

TEST_LIST = ['Test Receipt', 'Test Receipt 2']

# Test data with date objects
TEST_DATE = date(2023, 12, 25)
TEST_DATE_DF = pd.DataFrame({
    'date_col': [date(2023, 12, 25), date(2023, 12, 26)],
    'text_col': ['Test 1', 'Test 2']
})
TEST_DATE_DICT = {
    'date_field': date(2023, 12, 25),
    'text_field': 'Test Append'
}

old_data = [FileDataInfo(data_type="dataframe", snapshot="test_snapshot", original_file_name="test_df.csv")]

# URLs for testing
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1EZ8dMacJAPpVyKJrSOTQ3CG8mx1JhyWC-i0h9qurZNs/edit?gid=0#gid=0"
OFFICE_SHEET_URL = "https://onedrive.live.com/edit?id=D4064FF6F2B7F76C!105&resid=D4064FF6F2B7F76C!105&ithint=file%2cxlsx&ct=1733274618866&wdOrigin=OFFICECOM-WEB.START.EDGEWORTH&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=da84a250-8950-4855-b4d3-ae0d6b922633&wdo=2&cid=d4064ff6f2b7f76c"
OFFICE_SHEET_NAME = "Sheet2"

g_response = supabase.table('user_documents_access') \
    .select('refresh_token') \
    .match({'user_id': user_id, 'provider': 'google'}) \
    .execute()

if not g_response.data or len(g_response.data) == 0:
    print(f"No Google token found for user {user_id}")
google_refresh_token = g_response.data[0]['refresh_token']

ms_response = supabase.table('user_documents_access') \
    .select('refresh_token') \
    .match({'user_id': user_id, 'provider': 'microsoft'}) \
    .execute()

if not ms_response.data or len(ms_response.data) == 0:
    print(f"No Microsoft token found for user {user_id}")
ms_refresh_token = ms_response.data[0]['refresh_token']

# Add logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@pytest.fixture
def g_integration():
    """Fixture to create DocumentIntegrations instance with test credentials"""
    if not google_refresh_token:
        pytest.fail("GOOGLE_REFRESH_TOKEN environment variable not set")
        
    # Ensure other required environment variables are set
    required_vars = [
        'MS_TENANT_ID', 'MS_CLIENT_ID', 'MS_CLIENT_SECRET',
        'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.fail(f"Missing required environment variables: {', '.join(missing_vars)}")
            
    return GoogleIntegration(supabase, user_id)


@pytest.fixture
def msft_integration():
    """Fixture to create DocumentIntegrations instance with test credentials"""
    if not ms_refresh_token:
        pytest.fail("MS_REFRESH_TOKEN environment variable not set")
        
    # Ensure other required environment variables are set
    required_vars = [
        'MS_TENANT_ID', 'MS_CLIENT_ID', 'MS_CLIENT_SECRET',
        'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        pytest.fail(f"Missing required environment variables: {', '.join(missing_vars)}")
            
    return MicrosoftIntegration(supabase, user_id)


# Add these test cases after the existing fixtures

# async def test_google_integration_initialization(g_integration):
#     """Test that GoogleIntegration initializes correctly"""
#     assert g_integration is not None
#     assert g_integration.google_creds is not None

# async def test_append_to_current_google_sheet(g_integration):
#     """Test appending data to existing sheet"""
#     # Test with DataFrame
#     result = await g_integration.append_to_current_google_sheet(
#         TEST_DF,
#         GOOGLE_SHEET_URL,
#         "Receipts"  # Replace with your actual sheet name
#     )
#     assert result is True




# async def test_append_to_new_google_sheet(g_integration):
#     """Test creating and appending data to new sheet"""
#     # Test with DataFrame
#     result = await g_integration.append_to_new_google_sheet(
#         TEST_DF,
#         GOOGLE_SHEET_URL,
#         "TestNewSheet1"
#     )
#     assert result is True





async def test_extract_google_sheets_data(g_integration):
    """Test extracting data from Google Sheets"""
    df = await g_integration.extract_google_sheets_data(
        GOOGLE_SHEET_URL,
        "sd_grades"  # Replace with your actual sheet name
    )
    print("Extracted data:")
    print(df)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty



