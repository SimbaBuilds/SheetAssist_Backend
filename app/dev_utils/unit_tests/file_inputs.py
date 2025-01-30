# Replace the current sys.path modifications with this more robust approach
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import unittest
import asyncio
import io
from fastapi import UploadFile
import pandas as pd
from app.utils.preprocessing import FilePreprocessor
from pathlib import Path

class TestFileProcessors(unittest.TestCase):
    def setUp(self):
        self.preprocessor = FilePreprocessor()
        # Test file paths
        self.test_excel_path = os.path.join(os.path.dirname(__file__), "Lead Tracker.xlsx")
        self.test_csv_path = os.path.join(os.path.dirname(__file__), "receipts.csv")
        self.test_text_path = os.path.join(os.path.dirname(__file__), "Untitled document.txt")
        self.test_docx_path = os.path.join(os.path.dirname(__file__), "Untitled document.docx")

    def create_upload_file(self, content: bytes, filename: str, content_type: str) -> UploadFile:
        return UploadFile(
            file=io.BytesIO(content),
            filename=filename,
            headers={'content-type': content_type}
        )

    def test_process_excel(self):
        async def _test():
            # Test with file path
            df1 = await self.preprocessor.process_excel(self.test_excel_path)
            self.assertIsInstance(df1, pd.DataFrame)
            
            # Test with UploadFile
            with open(self.test_excel_path, 'rb') as f:
                content = f.read()
            upload_file = self.create_upload_file(content, "test.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
            # Process twice to verify file pointer handling
            df2 = await self.preprocessor.process_excel(upload_file)
            df3 = await self.preprocessor.process_excel(upload_file)
            
            self.assertIsInstance(df2, pd.DataFrame)
            self.assertIsInstance(df3, pd.DataFrame)
            # Verify dataframes are equal
            pd.testing.assert_frame_equal(df1, df2)
            pd.testing.assert_frame_equal(df2, df3)
        
        run_async_test(_test())

    # def test_process_csv(self):
    #     async def _test():
    #         # Test with file path
    #         df1 = await self.preprocessor.process_csv(self.test_csv_path)
    #         self.assertIsInstance(df1, pd.DataFrame)
            
    #         # Test with UploadFile
    #         with open(self.test_csv_path, 'rb') as f:
    #             content = f.read()
    #         upload_file = self.create_upload_file(content, "test.csv", "text/csv")
            
    #         # Process twice to verify file pointer handling
    #         df2 = await self.preprocessor.process_csv(upload_file)
    #         df3 = await self.preprocessor.process_csv(upload_file)
            
    #         self.assertIsInstance(df2, pd.DataFrame)
    #         self.assertIsInstance(df3, pd.DataFrame)
    #         pd.testing.assert_frame_equal(df1, df2)
    #         pd.testing.assert_frame_equal(df2, df3)
        
    #     run_async_test(_test())

    # def test_process_text(self):
    #     async def _test():
    #         # Test with file path
    #         text1 = await self.preprocessor.process_text(self.test_text_path)
    #         self.assertIsInstance(text1, str)
            
    #         # Test with UploadFile
    #         with open(self.test_text_path, 'rb') as f:
    #             content = f.read()
    #         upload_file = self.create_upload_file(content, "test.txt", "text/plain")
            
    #         # Process twice to verify file pointer handling
    #         text2 = await self.preprocessor.process_text(upload_file)
    #         text3 = await self.preprocessor.process_text(upload_file)
            
    #         self.assertIsInstance(text2, str)
    #         self.assertIsInstance(text3, str)
    #         self.assertEqual(text1, text2)
    #         self.assertEqual(text2, text3)
        
    #     run_async_test(_test())

    # def test_process_docx(self):
    #     async def _test():
    #         # Test with file path
    #         text1 = await self.preprocessor.process_docx(self.test_docx_path)
    #         self.assertIsInstance(text1, str)
            
    #         # Test with UploadFile
    #         with open(self.test_docx_path, 'rb') as f:
    #             content = f.read()
    #         upload_file = self.create_upload_file(content, "test.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
    #         # Process twice to verify file pointer handling
    #         text2 = await self.preprocessor.process_docx(upload_file)
    #         text3 = await self.preprocessor.process_docx(upload_file)
            
    #         self.assertIsInstance(text2, str)
    #         self.assertIsInstance(text3, str)
    #         self.assertEqual(text1, text2)
    #         self.assertEqual(text2, text3)
        
    #     run_async_test(_test())

def run_async_test(coro):
    return asyncio.get_event_loop().run_until_complete(coro)

if __name__ == '__main__':
    unittest.main()
