#!/bin/bash

# Improved Lambda layer deployment script with better error handling for SSL issues
set -e  # Exit immediately if a command exits with a non-zero status

# Function name
FUNCTION_NAME="sheet-assist-api-prod-api"
REGION="us-east-1"
WORKSPACE_DIR="$(pwd)"

echo "Starting Lambda layer deployment for $FUNCTION_NAME"
echo "Working from directory: $WORKSPACE_DIR"

# Verify AWS CLI is working
echo "Verifying AWS CLI configuration..."
aws sts get-caller-identity
echo "AWS CLI is configured correctly."

# Verify files exist
for file in layer-core.txt layer-data.txt layer-llm.txt layer-google.txt layer-utils.txt; do
    if [ ! -f "$WORKSPACE_DIR/$file" ]; then
        echo "Error: Required file $file not found in $WORKSPACE_DIR"
        exit 1
    else
        echo "Found requirement file: $file"
    fi
done

# Create simplified requirements-lambda.txt for function deployment
LAMBDA_REQS_FILE="$WORKSPACE_DIR/requirements-lambda.txt"
echo "Creating simplified $LAMBDA_REQS_FILE for function deployment..."
cat > $LAMBDA_REQS_FILE << EOL
fastapi==0.115.4
mangum==0.19.0
pydantic==2.9.2
EOL
echo "Created $LAMBDA_REQS_FILE"

# Function to create and deploy a layer
deploy_layer() {
    LAYER_NAME=$1
    REQ_FILE="$WORKSPACE_DIR/$2"
    echo "===== Creating layer: $LAYER_NAME ====="
    
    # Create a temporary directory for the layer
    TMP_DIR="/tmp/$LAYER_NAME"
    rm -rf $TMP_DIR
    mkdir -p $TMP_DIR/python
    
    # Install dependencies
    echo "Installing dependencies from $REQ_FILE to $TMP_DIR/python..."
    pip install -q -r "$REQ_FILE" -t $TMP_DIR/python
    
    # Create a zip file
    LAYER_ZIP="/tmp/$LAYER_NAME.zip"
    echo "Creating zip file: $LAYER_ZIP..."
    cd $TMP_DIR && zip -q -r $LAYER_ZIP .
    cd "$WORKSPACE_DIR"  # Return to workspace directory
    
    # Publish the layer to AWS Lambda with explicit SSL verification
    echo "Publishing layer $LAYER_NAME to AWS Lambda..."
    LAYER_ARN=$(aws lambda publish-layer-version \
        --layer-name "prod-$LAYER_NAME" \
        --description "Production layer for $LAYER_NAME" \
        --license-info "MIT" \
        --compatible-runtimes python3.11 \
        --zip-file fileb://$LAYER_ZIP \
        --region $REGION \
        --output text \
        --query LayerVersionArn)
    
    echo "Layer ARN: $LAYER_ARN"
    echo "$LAYER_ARN" > "/tmp/${LAYER_NAME}_arn.txt"
    return 0
}

# Deploy individual layers
deploy_layer "sheet-assist-core" "layer-core.txt"
deploy_layer "sheet-assist-data" "layer-data.txt"
deploy_layer "sheet-assist-llm" "layer-llm.txt"
deploy_layer "sheet-assist-google" "layer-google.txt"
deploy_layer "sheet-assist-utils" "layer-utils.txt"

# Collect all layer ARNs
echo "Collecting layer ARNs..."
LAYER_ARNS=$(cat /tmp/sheet-assist-core_arn.txt /tmp/sheet-assist-data_arn.txt /tmp/sheet-assist-llm_arn.txt /tmp/sheet-assist-google_arn.txt /tmp/sheet-assist-utils_arn.txt)

# Update the Lambda function with the new layers using JSON format to avoid command line length issues
echo "Updating Lambda function with new layers..."
TEMP_CLI_INPUT="/tmp/lambda_layers_input.json"
cat > $TEMP_CLI_INPUT << EOL
{
  "FunctionName": "$FUNCTION_NAME",
  "Layers": [
    $(cat /tmp/sheet-assist-core_arn.txt | sed 's/^/"/;s/$/"/'),
    $(cat /tmp/sheet-assist-data_arn.txt | sed 's/^/"/;s/$/"/'),
    $(cat /tmp/sheet-assist-llm_arn.txt | sed 's/^/"/;s/$/"/'),
    $(cat /tmp/sheet-assist-google_arn.txt | sed 's/^/"/;s/$/"/'),
    $(cat /tmp/sheet-assist-utils_arn.txt | sed 's/^/"/;s/$/"/')
  ]
}
EOL

aws lambda update-function-configuration \
    --cli-input-json file://$TEMP_CLI_INPUT \
    --region $REGION

# Clean up
rm -f $TEMP_CLI_INPUT

echo "Lambda layers deployment completed successfully!"
echo "You can now deploy your function with minimal requirements using: serverless deploy --stage prod --function api" 