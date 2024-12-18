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


TEST_LIST = ['Test Receipt', 'Test Receipt 2']




old_data = [FileDataInfo(data_type="dataframe", snapshot="test_snapshot", original_file_name="test_df.csv")]

# URLs for testing
GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/1EZ8dMacJAPpVyKJrSOTQ3CG8mx1JhyWC-i0h9qurZNs/edit?gid=2053550161#gid=2053550161"
GOOGLE_SHEET_NAME = "DuplicateTest"
OFFICE_SHEET_URL = "https://onedrive.live.com/edit?id=D4064FF6F2B7F76C!105&resid=D4064FF6F2B7F76C!105&ithint=file%2cxlsx&ct=1733274618866&wdOrigin=OFFICECOM-WEB.START.EDGEWORTH&wdPreviousSessionSrc=HarmonyWeb&wdPreviousSession=da84a250-8950-4855-b4d3-ae0d6b922633&wdo=2&cid=d4064ff6f2b7f76c"
OFFICE_SHEET_NAME = "Sheet1"


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


# @pytest.mark.asyncio
# async def test_google_sheets_append_existing_data(g_integration):
#     current_df = await g_integration.extract_google_sheets_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_NAME)
#     assert not current_df.empty, "Initial sheet should not be empty"
#     initial_row_count = len(current_df)
#     initial_values = current_df.iloc[0].to_dict()

#     # Attempt to append the same data
#     result = await g_integration.append_to_current_google_sheet(current_df.copy(), GOOGLE_SHEET_URL, GOOGLE_SHEET_NAME)
#     final_df = await g_integration.extract_google_sheets_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_NAME)

#     assert not final_df.empty, "Final sheet should not be empty"
#     assert len(final_df.columns) == len(current_df.columns), "Column count mismatch"
#     assert set(final_df.columns) == set(current_df.columns), "Column names mismatch"
#     assert len(final_df) == initial_row_count, "Row count mismatch"
#     assert final_df.iloc[0].to_dict() == initial_values, "First row changed unexpectedly"
#     assert result is True, "Append did not report success"

# @pytest.mark.asyncio
# async def test_google_sheets_append_mixed_data(g_integration):
#     logger.info("Starting test_google_sheets_append_mixed_data")
#     print("Starting test_google_sheets_append_mixed_data")
#     logger.info("Extracting current sheet data")
#     current_df = await g_integration.extract_google_sheets_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_NAME)
#     initial_row_count = len(current_df)
#     logger.info(f"Initial row count: {initial_row_count}")
#     print(f"Initial row count: {initial_row_count}")
    
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     logger.info(f"Using timestamp: {timestamp}")

#     logger.info("Creating test data")
#     print("Existing part created")
#     new_part = pd.DataFrame({
#         'Client First Name': [f'Test New 1_{timestamp}', f'Test New 2_{timestamp}'],
#         'Client Last Name': ['Smith', 'Jones'],
#         'Prescription Details': ['N/A', 'N/A'],
#         'Payment Type': ['Visa', 'Mastercard'],
#         'Payment Amount': ['100.00', '200.00'],
#         'Date ': ['03-13-2024', '03-13-2024'],
#         'Expiry Date': ['3-13-2025', '3-13-2025'],
#         'Other Notes': ['Test note 1', 'Test note 2']
#     })
#     print("New part created")
#     logger.info("Concatenating existing and new data")
#     mixed_data = pd.concat([current_df, new_part], ignore_index=True)
#     print("Mixed data created")
#     logger.info("Appending mixed data to sheet")
#     result = await g_integration.append_to_current_google_sheet(mixed_data, GOOGLE_SHEET_URL, GOOGLE_SHEET_NAME)
    
#     logger.info("Extracting final sheet data")
#     final_df = await g_integration.extract_google_sheets_data(GOOGLE_SHEET_URL, GOOGLE_SHEET_NAME)

#     expected_count = initial_row_count + 2
#     logger.info(f"Expected final row count: {expected_count}, Actual: {len(final_df)}")
#     print(f"Expected final row count: {expected_count}, Actual: {len(final_df)}")

#     assert not final_df.empty, "Final sheet should not be empty"
#     assert len(final_df) == expected_count, "Row count mismatch after append"
    
#     appended_rows = final_df[final_df['Client First Name'].str.contains(timestamp, na=False)]
#     logger.info(f"Found {len(appended_rows)} newly appended rows")
#     print(f"Found {len(appended_rows)} newly appended rows")
#     assert len(appended_rows) == 2, "Expected 2 new rows not found"
#     assert result is True, "Append did not report success"
    
#     logger.info("test_google_sheets_append_mixed_data completed successfully")
#     print("test_google_sheets_append_mixed_data completed successfully")

@pytest.mark.asyncio
async def test_office_excel_append_existing_data(msft_integration):
    logger.info("Starting test_office_excel_append_existing_data")
    current_df = await msft_integration.extract_msft_excel_data(OFFICE_SHEET_URL, OFFICE_SHEET_NAME)
    assert not current_df.empty, "Initial sheet should not be empty"
    initial_row_count = len(current_df)
    initial_values = current_df.iloc[0].to_dict()

    # Attempt to append the same data
    result = await msft_integration.append_to_current_office_sheet(current_df.copy(), OFFICE_SHEET_URL, OFFICE_SHEET_NAME)
    final_df = await msft_integration.extract_msft_excel_data(OFFICE_SHEET_URL, OFFICE_SHEET_NAME)

    assert not final_df.empty, "Final sheet should not be empty"
    assert len(final_df.columns) == len(current_df.columns), "Column count mismatch"
    assert set(final_df.columns) == set(current_df.columns), "Column names mismatch"
    assert len(final_df) == initial_row_count, "Row count mismatch"
    assert final_df.iloc[0].to_dict() == initial_values, "First row changed unexpectedly"
    assert result is True, "Append did not report success"

@pytest.mark.asyncio
async def test_office_excel_append_mixed_data(msft_integration):
    logger.info("Starting test_office_excel_append_mixed_data")
    
    # Step 1: Get current data and clean it up
    current_df = await msft_integration.extract_msft_excel_data(OFFICE_SHEET_URL, OFFICE_SHEET_NAME)
    
    # Debug: Print initial state
    print("\nCurrent DataFrame columns:")
    print(current_df.columns.tolist())
    
    # Get only the meaningful columns (non-empty column names)
    meaningful_columns = [col for col in current_df.columns if col.strip() != '']
    current_df = current_df[meaningful_columns]
    
    print("\nCleaned DataFrame columns:")
    print(current_df.columns.tolist())
    
    current_df = current_df.reset_index(drop=True)
    initial_row_count = len(current_df)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Step 2: Create new test data with matching columns
    new_part = pd.DataFrame({
        'Client First Name': [f'Test New 1_{timestamp}', f'Test New 2_{timestamp}'],
        'Client Last Name': ['Smith', 'Jones'],
        'Prescription Details': ['N/A', 'N/A'],
        'Payment Type': ['Visa', 'Mastercard'],
        'Payment Amount': ['100.00', '200.00'],
        'Date': ['03-13-2024', '03-13-2024'],
        'Expiry Date': ['3-13-2025', '3-13-2025'],
        'Other Notes': ['Test note 1', 'Test note 2']
    })
    
    # Ensure columns match exactly
    new_part = new_part[current_df.columns]
    
    # Step 3: Concatenate
    mixed_data = pd.concat([current_df, new_part], ignore_index=True)
    
    # Rest of the test
    result = await msft_integration.append_to_current_office_sheet(mixed_data, OFFICE_SHEET_URL, OFFICE_SHEET_NAME)
    final_df = await msft_integration.extract_msft_excel_data(OFFICE_SHEET_URL, OFFICE_SHEET_NAME)
    
    # Get only meaningful columns for final DataFrame too
    final_df = final_df[meaningful_columns]

    expected_count = initial_row_count + 2
    logger.info(f"Expected final row count: {expected_count}, Actual: {len(final_df)}")
    print(f"Expected final row count: {expected_count}, Actual: {len(final_df)}")

    assert not final_df.empty, "Final sheet should not be empty"
    assert len(final_df) == expected_count, "Row count mismatch after append"
    
    appended_rows = final_df[final_df['Client First Name'].str.contains(timestamp, na=False)]
    logger.info(f"Found {len(appended_rows)} newly appended rows")
    print(f"Found {len(appended_rows)} newly appended rows")
    assert len(appended_rows) == 2, "Expected 2 new rows not found"
    assert result is True, "Append did not report success"
    
    logger.info("test_office_excel_append_mixed_data completed successfully")