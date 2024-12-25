## Batch processing decision layer implementation 

## Overview: I am refactoring the main process_query endpoint to be renamed to process_query_standard, creating batch and status endpoints, and wrapping them all in a decision layer.  The batch processing endpoint is for when users enter a pdf or pdfs with more than 5 traditionally unreadable (modern Vision-AI readable) pages.  I am doing this for the sake of context window management for the vision models and the ability to create further specialized processors in the future.  Different user combinations of output preferences have different control flows as described below.

1. Refactoring 
    1. The current process_query endpoint will now be a subendpoint: process_query/standard.
    2. /process_query/batch will now handle the batch processing flow
    3. /status will chekc job status
    4. Full  structure:
    /process_query (Entry Point and Decision Router)
    ├── /standard (Current process_query logic)
    ├── /batch (New batch processing logic)
    └── /status (Batch job status checking)

2. Entry Point and Decision Router: /process_query
    1. Trigger
    Primary trigger: PDF page count > 5
    Routes to either standard (process_query_standard) or batch processing flow
    Returns appropriate response type based on processing path
    Creates unique job IDs for batch processes
    2. Job Management:
    Tracks overall job status and individual batch progress
    Stores intermediate results from each batch
    Manages job lifecycle (created → processing → completed/failed)
    Handles cleanup of completed/failed jobs
    3. Status Management:
    Tracks progress across all batches
    Maintains batch processing order
    Provides status updates for frontend polling
    Handles error states and partial completions


3. Batch Processing Flow 
    1. Varying processes based on user specified output preference:
        1. If online output and modify_existing:
            Decision Router → Batch Processor
                    ↓               ↓
                job_id        Process Batch 1
                    ↓               ↓
                Client ←----- Postprocess & Append
                    ↓               ↓
                Polls         Process Batch 2
                    ↓               ↓
                Status ←----- Postprocess & Append
                                    ⋮
            
            Can clear batch data after each append
            Real-time progress visible to user
        
        2. If online output and not modify_existing:
            Same as online output and modify_existing but we need to use append to new sheet on the first batch and append to current on the newly created sheet for the ensuing batches

        3. If download: 
            Decision Router → Batch Processor
                ↓               ↓
            job_id        Process Batch 1
                ↓               ↓
            Client        Store Batch 1 DF
                ↓               ↓
            Polls         Process Batch 2
                ↓               ↓
            Status        Store Batch 2 DF
                ↓               ⋮
            Status ←----- Combine All DFs
                ↓         Create Download
            Final File

            Must store all batch DataFrames
            Still shows processing progress
            Higher memory requirements
            Single combination step at end (can be csv, .xlsx, .txt, or .docx)
    2. File clean up should occur strategically throughout utilized FASTAPI background tasks and cleaner in @file_management 



Critical Schemas:

class QueryRequest(BaseModel):
    input_urls: Optional[List[InputUrl]] = []
    files_metadata: Optional[List[FileMetadata]] = []
    query: str
    output_preferences: OutputPreferences  # no longer Optional


class OutputPreferences(BaseModel):
    type: str  # 'download' or 'online' - no longer Optional
    destination_url: Optional[str] = None
    format: Optional[str] = None  # One of: 'csv', 'xlsx', 'docx', 'txt', 'pdf'
    modify_existing: Optional[bool] = None
    sheet_name: Optional[str] = None


class QueryResponse(BaseModel):
    """Unified response model for all query processing results"""
    result: TruncatedSandboxResult
    status: str  # "processing","completed" or "error"
    message: str  # Description of result or error message
    files: Optional[List[FileInfo]] = None  # For downloadable files
    num_images_processed: int = 0

    class Config:
        """docstring"""
        arbitrary_types_allowed = True