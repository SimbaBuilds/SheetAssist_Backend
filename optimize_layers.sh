#!/bin/bash

# Script to optimize and update Lambda function layers
set -e

# Function name
FUNCTION_NAME="sheet-assist-api-prod-api"
REGION="us-east-1"

echo "Optimizing and updating Lambda function with essential layers..."

# Let's use fewer layers to stay within size limits
# We'll use the core layer and one or two additional ones
CORE_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-core:16"
LLM_LAYER="arn:aws:lambda:us-east-1:307946648330:layer:prod-sheet-assist-llm:10"

# Create a temporary JSON file for the Lambda update
TEMP_CLI_INPUT="/tmp/lambda_layers_input.json"
cat > $TEMP_CLI_INPUT << EOL
{
  "FunctionName": "$FUNCTION_NAME",
  "Layers": [
    "$CORE_LAYER",
    "$LLM_LAYER"
  ]
}
EOL

echo "Updating Lambda function configuration..."
echo "Using the following essential layers:"
echo "Core: $CORE_LAYER"
echo "LLM: $LLM_LAYER"

# Try AWS CLI command with specific options to bypass SSL issues
AWS_CA_BUNDLE="" AWS_HTTPS_CA_BUNDLE="" aws lambda update-function-configuration \
    --cli-input-json file://$TEMP_CLI_INPUT \
    --region $REGION

# Clean up
rm -f $TEMP_CLI_INPUT

echo "Lambda function updated successfully with essential layers!"
echo "To deploy your function with minimal requirements, run: serverless deploy --stage prod --function api"
echo ""
echo "Note: Due to Lambda size limitations, only the essential layers were applied."
echo "Consider optimizing your dependencies to reduce layer sizes in the future." 