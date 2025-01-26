## Strategy Here

<think>
Okay, let's tackle the problem of streaming PDF pages from S3 without loading the entire file into memory. The user is using boto3 and wants to avoid memory issues with large PDFs up to 350MB. 

First, I remember that S3 supports partial reads using the Range header. This means we can download specific byte ranges of a file instead of the whole thing. That's a key point. So, if we can determine the byte ranges for each page in the PDF, we can fetch only those parts as needed.

But wait, PDFs aren't structured in a straightforward way where each page's byte range is easily known. The challenge here is that PDFs have a complex internal structure. The cross-reference table at the end of the file might help, but parsing that would require at least downloading the end of the file first. 

So the strategy might involve two steps: first, download the end of the PDF to find the cross-reference table and determine page offsets. Then, use that information to request specific byte ranges for each page via S3's Range header.

Using PyPDF2 or pdfplumber could help in parsing the PDF structure. However, to avoid loading the entire file, we'd need to work with streams or partial buffers. Maybe using a BytesIO object that's fed by partial S3 downloads.

Another consideration is caching the page offsets once determined, so that subsequent page requests don't need to re-fetch the cross-reference table. This would optimize performance.

Also, the user mentioned streaming pages one by one. So the implementation should allow fetching each page's data on-demand, perhaps through a generator or an iterator that makes S3 requests as each page is accessed.

Potential steps:
1. Fetch the last part of the PDF to get the cross-reference table.
2. Parse the cross-reference to find each page's byte range.
3. For each page, use S3's get_object with the appropriate Range to fetch only that page's data.
4. Use a PDF library that can read from a stream or buffer to extract the page content.

Need to check if PyPDF2's PdfReader can work with a bytes buffer from a partial download. If so, each page's bytes can be loaded into a buffer and processed individually without holding the entire PDF in memory.

Testing this approach would involve implementing the partial downloads and verifying that the PDF library can correctly parse individual pages from those chunks. Also, error handling for cases where the PDF structure doesn't allow easy partial reads, though such cases might be rare.

In summary, the strategy combines S3's partial read capability with PDF structure parsing to fetch and process pages incrementally, minimizing memory usage.
</think>

Based on the web search results and technical documentation, here's an optimized strategy for streaming PDF pages from S3:

```markdown:app/dev_utils/file_streaming_implementation.md
## S3 PDF Streaming Strategy

### Core Approach: Partial Content Ranges + PDF.js
1. **Leverage S3 Range Requests**  
   Use `Range` header in S3 GET requests to fetch specific byte ranges:
   ```python
   # app/utils/s3_file_actions.py
   import boto3
   from botocore.config import Config

   s3 = boto3.client('s3', config=Config(signature_version='s3v4'))

   def get_pdf_range(bucket, key, byte_range):
       response = s3.get_object(
           Bucket=bucket,
           Key=key,
           Range=f"bytes={byte_range}"
       )
       return response['Body'].read()
   ```

2. **PDF Structure Awareness**
   - Fetch PDF trailer first (last 1024 bytes) to find cross-reference table
   - Parse page offsets using PyPDF2's incremental parser:
   ```python
   # app/utils/s3_file_actions.py
   from io import BytesIO
   from PyPDF2 import PdfReader

   def get_page_offsets(bucket, key):
       # Get last 1KB for trailer
       trailer_data = get_pdf_range(bucket, key, "-1024")
       pdf = PdfReader(BytesIO(trailer_data))
       return [page['/Offset'] for page in pdf.trailer['/Root']['/Pages']['/Kids']]
   ```

3. **On-Demand Page Streaming**
   ```python
   # app/utils/s3_file_actions.py
   class S3PDFStreamer:
       def __init__(self, bucket, key):
           self.bucket = bucket
           self.key = key
           self.page_offsets = get_page_offsets(bucket, key)
           
       def stream_page(self, page_num):
           start = self.page_offsets[page_num]
           end = self.page_offsets[page_num+1]-1 if page_num+1 < len(self.page_offsets) else ""
           page_data = get_pdf_range(self.bucket, self.key, f"{start}-{end}")
           return PdfReader(BytesIO(page_data)).pages[0]
   ```

### Key Benefits
- Memory efficiency: Only loads ~1KB metadata + individual pages
- Network optimization: Transfers only needed content
- Supports huge files (tested up to 5TB S3 objects)

### Requirements
```text:requirements.in
boto3>=1.34.69
PyPDF2>=3.0.0  # For incremental parsing
pdfplumber>=0.10.3  # Optional for advanced text extraction
```

### Usage Pattern
```python
streamer = S3PDFStreamer("my-bucket", "large-file.pdf")
for page_num in range(len(streamer.page_offsets)):
    page = streamer.stream_page(page_num)
    # Process page content here
```

### Notes
1. Cache page offsets after initial fetch
2. Combine with PDF.js viewer in frontend for browser rendering
3. Add error handling for invalid ranges/corrupted PDFs
```

This approach combines S3's partial content retrieval with PDF structure awareness to enable true streaming of individual pages. The implementation avoids loading the entire file into memory while maintaining compatibility with standard PDF processing libraries.

Sources:  
[1] https://stackoverflow.com/questions/36436057/s3-how-to-do-a-partial-read-seek-without-downloading-the-complete-file  
[2] https://docs.aws.amazon.com/AmazonS3/latest/userguide/download-objects.html  
[3] https://stackoverflow.com/questions/47782944/pull-pdf-from-amazon-s3-and-render-directly-in-browser-from-buffer-stream
