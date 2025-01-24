## S3 Implementation - Transitioning from Local File Storage to Amazon S3

### Overview
Transitioning from local file storage to S3-based storage to handle large files (up to 350MB) efficiently using streaming approaches.

### Core Components

1. **S3 File Management (`s3_file_management.py`)**
   - Handles temporary file lifecycle in S3
   - Manages cleanup of old files
   - Provides session-based file organization
   - Environment variables needed:
     ```
     AWS_TEMP_BUCKET=your-temp-bucket-name
     AWS_ACCESS_KEY_ID=your-access-key
     AWS_SECRET_ACCESS_KEY=your-secret-key
     AWS_REGION=your-region
     ```

2. **S3 File Actions (`s3_file_actions.py`)**
   - Handles streaming uploads and downloads
   - Provides file access patterns for different file types
   - Manages direct S3 streaming capabilities
   - Key functions:
     - `stream_upload`: Stream files directly to S3
     - `get_streaming_body`: Get streaming access to S3 files
     - `stream_download`: Stream files from S3
     - `get_file_range`: Get specific byte ranges for large files

### Implementation Notes

1. **File Lifecycle Management**
   - 24-hour retention period for temporary files
   - Automatic cleanup via S3 lifecycle rules
   - Immediate cleanup after successful processing
   - Cleanup triggers for abandoned sessions

2. **Error Handling Strategy**
   - Implement exponential backoff retry (3 attempts)
   - Clear error messaging to frontend
   - Cleanup partial uploads on failure
   - Transaction-like approach for multi-step operations

3. **Performance Considerations**
   - Streaming-first approach for large files
   - No caching layer (files are temporary/single-use)
   - Async processing where applicable
   - Optimized for 350MB file handling

4. **Security Implementation**
   - Server-side encryption (SSE-S3)
   - Pre-signed URLs for direct client access
   - Strict bucket policies
   - Session-based access control

5. **Migration Strategy**
   - Full immediate migration
   - Simultaneous update of all dependent modules
   - Rollback capability included
   - No backwards compatibility needed

### Functionality Note: This backend python project receives front end payloads that involve user requests, uploaded files, and online spreadhseet url information.  This data is preprocessed via Python and multimodal large language models, passed to an LLM operating in a sandbox code execution environment, and postprocessed.  Postprocessing either involves appending the result to a user's online spreadsheet or serving the new data for download.  Keep this functionality in mind during your implementation. 

### Implementation Steps

1. **S3 Bucket Setup**
   - Create S3 bucket for temporary files
   - Configure CORS for direct uploads
   - Set up lifecycle rules for automatic cleanup
   - Configure bucket policies for security

2. **Code Migration: Integrate with s3_file_actions.py**
   - [x] Create `s3_file_actions.py` for streaming operations
   - [ ] Modify process_query.py endpoints to use S3
   - [ ] Modify data_visualization.py endpoints to use S3
   - [ ] Update preprocessing.py to use S3 streams
   - [ ] Update llm_service.py to use S3 streams
   - [ ] Update postprocessing.py for S3 output handling
   - [ ] Update download.py for S3 output handling

2. **Code Migration 2: Integrate with s3_file_management.py**
   - [x] Create `s3_file_management.py` for temp file management
   - [ ] Modify process_query.py endpoints 
   - [ ] Modify data_visualization.py endpoint
   - [ ] Update preprocessing.py 
   - [ ] Update llm_service.py 
   - [ ] Update postprocessing.py 
   - [ ] Update download.py




