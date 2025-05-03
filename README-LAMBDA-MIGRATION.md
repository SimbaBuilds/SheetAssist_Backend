# AWS Lambda Migration Guide for SheetAssist API

This document outlines the migration of the SheetAssist API from AWS Elastic Beanstalk to AWS Lambda.

## Migration Status

✅ The migration has been successfully completed. The API is now running on AWS Lambda with API Gateway.

## Lambda Endpoint

The API is accessible at: https://pnd0g25gj2.execute-api.us-east-1.amazonaws.com/

## Key Files

- `lambda_handler.py`: The main Lambda handler that interfaces with your FastAPI application
- `serverless.yml`: Configuration for the Serverless Framework
- `env-dev.yml` and `env-prod.yml`: Environment variables for different stages
- `requirements-lambda.txt`: Optimized dependencies for Lambda
- `resources.yml`: AWS CloudFormation templates for additional resources
- `deploy.sh`: Deployment script

## Notable Configuration

- **Timeout**: Set to 15 minutes (900 seconds) as requested
- **Memory**: 1024MB (configurable in serverless.yml)
- **Python Runtime**: 3.11
- **S3 Bucket**: Automatically created for temporary file storage with 1-day expiration
- **Logging**: CloudWatch logs configured for both Lambda and API Gateway

## Deployment Instructions

1. Ensure AWS credentials are configured:
   ```
   aws configure
   ```

2. Update environment variables in `env-dev.yml` or `env-prod.yml` with your actual credentials and settings

3. Deploy the application:
   ```
   ./deploy.sh dev    # For development environment
   ./deploy.sh prod   # For production environment
   ```

4. To remove deployment:
   ```
   ./deploy.sh dev --remove
   ```

## Important Notes

- **Environment Variables**: All necessary environment variables have been added to the environment configuration files. Replace placeholder values with actual credentials before deploying.
- **Cleanup**: The S3 bucket has an automatic 1-day expiration policy for all files.
- **Cold Starts**: Lambda functions have cold starts. The first invocation may be slower.

## Benefits of Lambda

- **Cost Efficiency**: Pay only for compute time used, not for idle servers
- **Automatic Scaling**: Scales automatically with traffic
- **Reduced Maintenance**: No server management required
- **High Availability**: Built-in redundancy and fault tolerance

## Troubleshooting

If you encounter issues:

1. Check CloudWatch logs for error messages
2. Verify environment variables are correctly set
3. Ensure all required dependencies are in requirements-lambda.txt
4. For local testing, run `python lambda_handler.py`

## Local Development

For local development and testing:

1. Create a `.env` file with your environment variables
2. Run `python lambda_handler.py` which will start a local server using uvicorn
3. Test your API at http://localhost:8000

## Next Steps

- [ ] Update CI/CD pipelines for Lambda deployment
- [ ] Set up monitoring and alarms in CloudWatch
- [ ] Configure custom domain in API Gateway
- [ ] Implement API Gateway caching if needed
- [ ] Set up AWS X-Ray for tracing (optional) 