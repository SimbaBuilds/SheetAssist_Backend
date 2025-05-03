# Lambda Migration Checklist

## Before Deployment
- [ ] Update `env-dev.yml` and `env-prod.yml` with actual values for:
  - [ ] Supabase credentials
  - [ ] Google API credentials
  - [ ] Microsoft API credentials
  - [ ] Anthropic API key
  - [ ] OpenAI API key
  - [ ] AWS S3 bucket names

- [ ] Verify `requirements-lambda.txt` contains all necessary dependencies
- [ ] Test locally by running `python lambda_handler.py`

## Deployment
- [ ] Run deployment for dev environment: `./deploy.sh dev`
- [ ] Verify API endpoints are working at the provided URL
- [ ] Test critical endpoints:
  - [ ] Health check endpoint (`/health`)
  - [ ] Sheet names endpoint
  - [ ] Query processing endpoint
  - [ ] Download endpoint
  - [ ] Data visualization endpoint

## Post-Deployment
- [ ] Set up CloudWatch alarms for:
  - [ ] Error rates
  - [ ] Latency
  - [ ] Invocation count
  - [ ] Duration
- [ ] Update client-side configurations to use the new API endpoint
- [ ] Monitor logs for any errors or issues
- [ ] Set up custom domain in API Gateway (optional)
- [ ] Update CI/CD pipelines for Lambda deployment

## Rollback Plan (if needed)
- [ ] Keep original Elastic Beanstalk environment active during testing period
- [ ] If issues arise, route traffic back to Elastic Beanstalk endpoint
- [ ] Troubleshoot Lambda issues before attempting redeployment

## Final Step
- [ ] Once stable, decommission Elastic Beanstalk environment to save costs 