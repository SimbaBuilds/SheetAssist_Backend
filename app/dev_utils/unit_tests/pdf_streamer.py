import unittest
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from pypdf import PdfReader
from app.utils.s3_file_actions import S3PDFStreamer
import boto3

class TestS3PDFStreamer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.bucket = 'sheet-assist-temp-bucket'
        cls.key = 'test_files/simple_text.pdf'
        cls.s3 = boto3.client('s3')
        
        # Create a simple PDF with text content
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Add three pages with text
        for i in range(3):
            c.drawString(100, 700, f"This is page {i+1} with some text content.")
            c.drawString(100, 680, "Additional text to ensure content is present.")
            c.showPage()
        
        c.save()
        buffer.seek(0)
        pdf_content = buffer.read()
        
        # Upload to S3
        cls.s3.put_object(
            Bucket=cls.bucket,
            Key=cls.key,
            Body=pdf_content
        )
        
        # Store size and page count for tests
        cls.pdf_size = len(pdf_content)
        cls.page_count = 3

    def test_stream_single_page(self):
        """Test streaming a single page."""
        streamer = S3PDFStreamer(self.s3, self.bucket, self.key)
        page_data = streamer.stream_page(1)
        
        # Verify the page content
        reader = PdfReader(BytesIO(page_data))
        self.assertEqual(len(reader.pages), 1)
        self.assertIsNotNone(reader.pages[0])

    def test_full_document_stream(self):
        """Test streaming the entire document."""
        streamer = S3PDFStreamer(self.s3, self.bucket, self.key)
        pages = list(streamer.stream_full_document())
        
        # Verify we got all pages
        self.assertEqual(len(pages), self.page_count)
        
        # Verify each page is a valid PDF
        for page_data in pages:
            reader = PdfReader(BytesIO(page_data))
            self.assertEqual(len(reader.pages), 1)
            self.assertIsNotNone(reader.pages[0])

    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        cls.s3.delete_object(Bucket=cls.bucket, Key=cls.key)

if __name__ == '__main__':
    unittest.main()
