#!/bin/bash

# Script to update Lambda function with the successfully uploaded layers
set -e

# Function name
FUNCTION_NAME="sheet-assist-api-prod-api"
REGION="us-east-1"

echo "Updating Lambda function with layers..."

# Hardcode the layer ARNs that were successfully uploaded
CORE_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-core:16"
DATA_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-data:12"
LLM_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-llm:10"
GOOGLE_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-google:10"
# Let's use the previous version of utils layer since the new one failed
UTILS_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-utils:9"

# Create a temporary JSON file for the Lambda update
TEMP_CLI_INPUT="/tmp/lambda_layers_input.json"
cat > $TEMP_CLI_INPUT << EOL
{
  "FunctionName": "$FUNCTION_NAME",
  "Layers": [
    "$CORE_LAYER",
    "$DATA_LAYER",
    "$LLM_LAYER", 
    "$GOOGLE_LAYER",
    "$UTILS_LAYER"
  ]
}
EOL

echo "Updating Lambda function configuration..."
echo "Using the following layers:"
echo "Core: $CORE_LAYER"
echo "Data: $DATA_LAYER"
echo "LLM: $LLM_LAYER"
echo "Google: $GOOGLE_LAYER"
echo "Utils: $UTILS_LAYER"

# Try AWS CLI command with specific options to bypass SSL issues
AWS_CA_BUNDLE="" AWS_HTTPS_CA_BUNDLE="" aws lambda update-function-configuration \
    --cli-input-json file://$TEMP_CLI_INPUT \
    --region $REGION

# Clean up
rm -f $TEMP_CLI_INPUT

echo "Lambda function updated successfully!"
echo "You can now deploy your function with minimal requirements using: serverless deploy --stage prod --function api" 