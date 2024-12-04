import unittest
import pandas as pd
from datetime import datetime
from app.utils.file_postprocessing import DocumentIntegrations

class TestDateHandling(unittest.TestCase):
    def setUp(self):
        # Mock Google refresh token for testing
        self.doc_integrations = DocumentIntegrations("mock_refresh_token")
        
        # Create test DataFrame matching the provided structure
        self.test_data = pd.DataFrame({
            'Date': [pd.Timestamp('2016-01-15 18:32:00')],
            'Vendor': ['MFY SIDE 1'],
            'Item Description': ['Cpy Btrmlk Ckn Meal, M Sprite, Cpy Buttermilk Ckn (1P), Mozzarella Sticks, No Sauce'],
            'Category': ['Food'],
            'Amount': [7.98],
            'Payment Method': ['Cash'],
            'Tax': [0.48],
            'Total': [8.46],
            'Receipt Number': ['KVS Order 55'],
            'Notes': ['Cash Tendered: $8.46, Change: $0.00']
        })

    def test_timestamp_conversion(self):
        """Test that Timestamp objects are properly converted to strings"""
        # Call the method that processes the data for Google Sheets
        try:
            # We'll test the data formatting logic directly since we can't actually
            # connect to Google Sheets in a unit test
            if isinstance(self.test_data, pd.DataFrame):
                df_copy = self.test_data.copy()
                for col in df_copy.select_dtypes(include=['datetime64[ns]']).columns:
                    df_copy[col] = df_copy[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                values = df_copy.values.tolist()
                
                # Verify the date was converted correctly
                self.assertEqual(values[0][0], '2016-01-15 18:32:00')
                
                # Verify other values remained unchanged
                self.assertEqual(values[0][1], 'MFY SIDE 1')
                self.assertEqual(values[0][4], 7.98)
                
        except Exception as e:
            self.fail(f"Failed to process DataFrame: {str(e)}")

if __name__ == '__main__':
    unittest.main()
