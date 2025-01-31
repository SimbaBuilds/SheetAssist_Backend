// import { AxiosResponse } from 'axios';
// import api from './api';
// import { OutputPreferences, FileUploadMetadata, QueryRequest, QueryResponse, InputUrl, ProcessingState } from '@/lib/types/dashboard';
// import { AcceptedMimeType, MIME_TYPES } from '@/lib/constants/file-types';
// import { createClient } from '@/lib/supabase/client';
// import { isUserOnProPlan, getUserSubscriptionId, trackUsage } from '@/lib/stripe/usage'
// import { PLAN_REQUEST_LIMITS, PLAN_IMAGE_LIMITS } from '@/lib/constants/pricing'
// import { logRequest } from '@/lib/services/loggers/request-logger';
// import { logError } from '@/lib/services/loggers/error-logger';
// import { uploadFileToS3 } from '@/lib/s3/s3-upload';
// import { S3_SIZE_THRESHOLD } from '@/lib/constants/file-types';




// // Helper function to update user usage statistics
// async function updateUserUsage(userId: string, success: boolean, numImagesProcessed: number = 0) {
//   // console.log('🔍 [process_query] Starting updateUserUsage:', { userId, success, numImagesProcessed });
//   const supabase = createClient();
  
//   // Get current usage data
//   const { data: usageData, error: usageError } = await supabase
//     .from('user_usage')
//     .select('*')
//     .eq('user_id', userId)
//     .single();

//   if (usageError) {
//     console.error('Error fetching user usage:', usageError);
//     return;
//   }

//   const newRequestCount = (usageData?.requests_this_month || 0) + (success ? 1 : 0);
//   const newImageCount = (usageData?.images_processed_this_month || 0) + (success ? numImagesProcessed : 0);
//   const imagesToLog = newImageCount - (usageData?.images_processed_this_month || 0);
//   // console.log('🔍 [process_query] Current usage counts:', { 
//   //   newRequestCount, 
//   //   newImageCount, 
//   //   currentRequests: usageData?.requests_this_month,
//   //   currentImages: usageData?.images_processed_this_month 
//   // });

//   const updateData = {
//     requests_this_month: newRequestCount,
//     images_processed_this_month: newImageCount,
//     requests_this_week: (usageData?.requests_this_week || 0) + (success ? 1 : 0),
//     requests_previous_3_months: (usageData?.requests_previous_3_months || 0) + (success ? 1 : 0),
//     unsuccessful_requests_this_month: (usageData?.unsuccessful_requests_this_month || 0) + (success ? 0 : 1)
//   };

//   // Update usage statistics
//   await supabase
//     .from('user_usage')
//     .update(updateData)
//     .eq('user_id', userId);

//   // Check if pro user and handle overages
//   if (success) {
//     const isProUser = await isUserOnProPlan(userId);
//     if (isProUser) {
//       const subscriptionId = await getUserSubscriptionId(userId);
//       if (subscriptionId) {
//         // Track processing usage if over limit
//         if (newRequestCount > PLAN_REQUEST_LIMITS.pro) {
//           // console.log(`[process_query] Tracking processing usage overage for user ${userId}. Count: ${newRequestCount}`);
//           try {
//             await trackUsage({
//               subscriptionId,
//               type: 'processing',
//               quantity: newRequestCount,
//               imagesToLog,
//               userId
//             });
//             // console.log(`[process_query] Successfully tracked processing usage for user ${userId}`);
//           } catch (error) {
//             console.error(`[process_query] Failed to track processing usage for user ${userId}:`, error);
//           }
//         }
        
//         // Track images usage if over limit
//         if (newImageCount > PLAN_IMAGE_LIMITS.pro) {
//           // console.log(`[process_query] Tracking image usage overage for user ${userId}. Count: ${newImageCount}`);
//           try {
//             await trackUsage({
//               subscriptionId,
//               type: 'images',
//               quantity: newImageCount,
//               imagesToLog,
//               userId
//             });
//             // console.log(`[process_query] Successfully tracked image usage for user ${userId}`);
//           } catch (error) {
//             console.error(`[process_query] Failed to track image usage for user ${userId}:`, error);
//           }
//         }
//       }
//     }
//   }
// }

// class QueryService {
//   // Increase polling interval to reduce server load
//   private readonly POLLING_INTERVAL = 5000; // 5 seconds
//   private readonly MAX_RETRIES = 15;
  
//   // Add new timeout constants
//   private readonly BATCH_TIMEOUT = 7200000; // 2 hours for batch processes
//   private readonly STANDARD_TIMEOUT = 600000; // 10 minutes
//   private readonly POLLING_TIMEOUT = 180000;   // 3 minutes

//   private async pollJobStatus(
//     jobId: string,
//     userId: string,
//     query: string,
//     signal?: AbortSignal,
//     onProgress?: (state: ProcessingState) => void
//   ): Promise<QueryResponse> {
//     const params = new URLSearchParams();
//     params.append('job_id', jobId);
//     let retries = 0;

//     const startTime = Date.now();
//     const MAX_TOTAL_TIME = 3600000; // 1 hour max total polling time

//     console.log('[process_query] Starting status polling for job:', jobId);

//     while (true) {
//       if (Date.now() - startTime > MAX_TOTAL_TIME) {
//         const errorState: ProcessingState = {
//           status: 'error',
//           message: 'Maximum polling time exceeded'
//         };
//         onProgress?.(errorState);
//         return {
//           status: 'error',
//           message: errorState.message,
//           num_images_processed: 0
//         };
//       }

//       if (signal?.aborted) {
//         console.log('[process_query] Polling aborted by signal');
//         return {
//           status: 'canceled',
//           message: 'Request was canceled',
//           num_images_processed: 0
//         };
//       }

//       try {
//         console.log('[process_query] Polling status...');
//         const response = await api.post('/process_query/status', params, {
//           headers: {
//             'Content-Type': 'application/x-www-form-urlencoded'
//           },
//           signal,
//           timeout: this.POLLING_TIMEOUT,
//         });

//         const result = response.data;
        
//         // Important: Check for error indicators in both status and message
//         const hasError = 
//           result.status === 'error' || 
//           result.error || 
//           (result.message && result.message.toLowerCase().includes('error'));

//         if (hasError) {
//           const errorMessage = result.error || result.message || 'Backend processing error';
//           await logError({
//             userId,
//             originalQuery: query,
//             message: result.message || 'Backend processing error',
//             errorCode: 'BACKEND_ERROR',
//             requestType: 'query',
//             startTime
//           });
          
//           const errorState: ProcessingState = {
//             status: 'error',
//             message: errorMessage
//           };
//           onProgress?.(errorState);
          
//           return {
//             status: 'error',
//             message: errorMessage,
//             num_images_processed: result.num_images_processed || 0,
//             error: errorMessage
//           };
//         }

//         // Create processing state from result
//         const processingState: ProcessingState = {
//           status: result.status,
//           message: result.message || `Processing page ${result.num_images_processed || 0} of ${result.total_pages || '?'}`,
//           progress: result.num_images_processed ? {
//             processed: result.num_images_processed,
//             total: result.total_pages || null
//           } : undefined
//         };

//         // Update progress for non-error states
//         onProgress?.(processingState);

//         if (result.status === 'completed') {
//           return result;
//         }

//         // Only continue polling if status is 'processing' or 'created'
//         if (result.status !== 'processing' && result.status !== 'created') {
//           const errorState: ProcessingState = {
//             status: 'error',
//             message: `Unexpected status: ${result.status}`
//           };
//           onProgress?.(errorState);
//           return {
//             status: 'error',
//             message: errorState.message,
//             num_images_processed: result.num_images_processed || 0
//           };
//         }

//         const backoffTime = Math.min(
//           this.POLLING_INTERVAL * Math.pow(1.5, retries),
//           15000
//         );
        
//         await new Promise(resolve => setTimeout(resolve, backoffTime));
//         continue;

//       } catch (error: unknown) {
//         const typedError = error as { code?: string; message?: string };
//         console.error('[process_query] Polling error:', {
//           code: typedError?.code,
//           message: typedError?.message,
//           retries
//         });

//         // Handle connection timeouts
//         if (typedError?.code === 'ECONNABORTED') {
//           retries++;
//           if (retries >= this.MAX_RETRIES) {
//             const errorState: ProcessingState = {
//               status: 'error',
//               message: 'Connection timeout',

//             };
//             onProgress?.(errorState);
//             return {
//               status: 'error',
//               message: errorState.message,
//               num_images_processed: 0
//             };
//           }
//           // Wait longer between retries on timeout
//           await new Promise(resolve => setTimeout(resolve, this.POLLING_INTERVAL * 2));
//           continue;
//         }

//         // For other errors
//         retries++;
//         if (retries >= this.MAX_RETRIES) {
//           const errorState: ProcessingState = {
//             status: 'error',
//             message: 'Maximum retry attempts exceeded',
//             details: 'There was a problem processing your request. Please try again later.'
//           };
//           onProgress?.(errorState);
//           return {
//             status: 'error',
//             message: 'Maximum retry attempts exceeded. Please try rephrasing your request or breaking it down into multiple steps.',
//             num_images_processed: 0
//           };
//         }
//         await new Promise(resolve => setTimeout(resolve, this.POLLING_INTERVAL));
//       }
//     }
//   }

//   async processQuery(
//     query: string,
//     webUrls: InputUrl[] = [],
//     files?: File[],
//     outputPreferences?: OutputPreferences,
//     signal?: AbortSignal,
//     onProgress?: (state: ProcessingState) => void
//   ): Promise<QueryResponse> {
//     const startTime = Date.now();
//     const supabase = createClient();
//     const user = await supabase.auth.getUser();
//     const userId = user.data.user?.id;

//     if (!userId) {
//       throw new Error('User not authenticated');
//     }

//     const formData = new FormData();
    
//     // Handle file uploads and metadata creation
//     const filesMetadata: FileUploadMetadata[] = [];
//     const fileUploads: Promise<void>[] = [];
    
//     if (files?.length) {
//       files.forEach((file, originalIndex) => {
//         const metadata: FileUploadMetadata = {
//           name: file.name,
//           type: file.type as AcceptedMimeType,
//           extension: `.${file.name.split('.').pop()?.toLowerCase() || ''}`,
//           size: file.size,
//           index: originalIndex  // Keep original index in metadata
//         };

//         const isImage = file.type === MIME_TYPES.PNG || file.type === MIME_TYPES.JPEG || file.type === MIME_TYPES.JPG;
//         if (file.size >= S3_SIZE_THRESHOLD || isImage) {
//           // Large file or image - upload to S3
//           console.log('[process_query] File will be uploaded to S3', {
//             fileName: file.name,
//             fileSize: file.size,
//             fileType: file.type,
//             reason: file.size >= S3_SIZE_THRESHOLD ? 'size' : 'image type',
//             userId
//           });
          
//           const uploadPromise = uploadFileToS3({
//             name: file.name,
//             type: file.type,
//             size: file.size,
//             file
//           }, userId)
//             .then(result => {
//               console.log('[process_query] S3 upload completed successfully', {
//                 fileName: file.name,
//                 s3Key: result.key,
//                 userId
//               });
//               metadata.s3_key = result.key;
//               metadata.s3_url = result.url;
//             })
//             .catch(error => {
//               console.error('[process_query] S3 upload failed', {
//                 fileName: file.name,
//                 error,
//                 userId
//               });
//               throw error;
//             });
//           fileUploads.push(uploadPromise);
//         } else {
//           console.log('[process_query] File below S3 threshold, including in form data', {
//             fileName: file.name,
//             fileSize: file.size,
//             userId
//           });
//           // Small file - append to formData directly with 'files' key
//           formData.append('files', file);
//         }

//         filesMetadata.push(metadata);
//       });
//     }

//     // Wait for all S3 uploads to complete
//     if (fileUploads.length > 0) {
//       await Promise.all(fileUploads);
//     }

//     // Part 1: JSON payload with metadata
//     const jsonData: QueryRequest = {
//       query,
//       input_urls: webUrls,
//       files_metadata: filesMetadata,
//       output_preferences: outputPreferences
//     };
//     formData.append('json_data', JSON.stringify(jsonData));

//     try {
//       // Update initial processing state
//       onProgress?.({
//         status: 'processing',
//         message: 'Processing your request...'
//       });

//       const response: AxiosResponse<QueryResponse> = await api.post('/process_query', formData, {
//         headers: { 
//           'Content-Type': 'multipart/form-data'
//         },
//         maxBodyLength: Infinity,
//         maxContentLength: Infinity,
//         signal,
//         timeout: this.STANDARD_TIMEOUT,
//       });

//       const initialResult: QueryResponse = response.data;

//       // If this is a batch process, handle polling
//       if (initialResult.job_id) {
//         // Update timeout for batch processing
//         api.defaults.timeout = this.BATCH_TIMEOUT;
        
//         try {
//           const result = await this.pollJobStatus(
//             initialResult.job_id,
//             userId,
//             query,
//             signal,
//             onProgress
//           );

//           // Handle the polling result
//           if (result.status === 'error' || result.status === 'canceled') {
//             await logRequest({
//               userId,
//               query,
//               fileMetadata: filesMetadata,
//               inputUrls: webUrls,
//               startTime,
//               status: result.status,
//               success: false,
//               errorMessage: result.message,
//               requestType: 'query',
//               numImagesProcessed: result.num_images_processed
//             });
//             return result;
//           }

//           await logRequest({
//             userId,
//             query,
//             fileMetadata: filesMetadata,
//             inputUrls: webUrls,
//             startTime,
//             status: 'completed',
//             success: true,
//             requestType: 'query',
//             numImagesProcessed: result.num_images_processed
//           });
//           await updateUserUsage(userId, true, result.num_images_processed || 0);
//           return result;  // Return the final result
//         } catch (error) {
//           // This should rarely happen now as errors are handled in pollJobStatus
//           console.error('Unexpected error during polling:', error);
//           const errorState: ProcessingState = {
//             status: 'error',
//             message: 'An unexpected error occurred',
//             details: 'Please try again later or contact support if the problem persists.'
//           };
//           onProgress?.(errorState);
//           return {
//             status: 'error',
//             message: errorState.message,
//             num_images_processed: 0
//           };
//         } finally {
//           // Reset timeout to standard
//           api.defaults.timeout = this.STANDARD_TIMEOUT;
//         }
//       } else {
//         // Handle non-batch processing result
//         if (initialResult.status === 'error') {
//           // await Promise.all([
//           //   logError({
//           //     userId,
//           //     originalQuery: query,
//           //     fileNames: files?.map(f => f.name),
//           //     docNames: webUrls.map(url => url.url),
//           //     message: initialResult.message || 'Processing error',
//           //     errorCode: 'PROCESSING_ERROR',
//           //     requestType: 'query',
//           //     startTime
//           //   })
//           // ]);
//           throw new Error(initialResult.message || 'Processing failed');
//         }

//         await logRequest({
//           userId,
//           query,
//           fileMetadata: filesMetadata,
//           inputUrls: webUrls,
//           startTime,
//           status: initialResult.status || 'completed',
//           success: initialResult.status === 'completed',
//           requestType: 'query',
//           numImagesProcessed: initialResult.num_images_processed
//         });
//         await updateUserUsage(userId, true, initialResult.num_images_processed || 0);
//         return initialResult;
//       }

//     } catch (error: unknown) {
//       const typedError = error as { 
//         response?: { 
//           data?: { message?: string }, 
//           status?: number 
//         }, 
//         message?: string, 
//         code?: string, 
//         stack?: string,
//         name?: string 
//       };
//       const errorMessage = typedError?.response?.data?.message || 
//                          typedError?.message || 
//                          'An unexpected error occurred';
      
//       onProgress?.({
//         status: 'error',
//         message: errorMessage
//       });

//       // Log error to both tables
//       const errorCode = typedError?.response?.status?.toString() || typedError?.code || 'UNKNOWN';
//       await Promise.all([
//         // Log to error_log table
//         logError({
//           userId,
//           originalQuery: query,
//           fileNames: files?.map(f => f.name),
//           docNames: webUrls.map(url => url.url),
//           message: errorMessage,
//           errorCode,
//           requestType: 'query',
//           errorMessage: typedError?.stack,
//           startTime
//         }),
//         // Log to request_log table (existing)
//         logRequest({
//           userId,
//           query,
//           fileMetadata: filesMetadata,
//           inputUrls: webUrls,
//           startTime,
//           status: (typedError?.response?.status?.toString() || typedError?.name || 'error') as string,
//           success: false,
//           errorMessage,
//           requestType: 'query'
//         })
//       ]);

//       // Update user usage statistics for failed request
//       await updateUserUsage(userId, false);

//       throw error;
//     }
//   }
// }

// export const queryService = new QueryService();



