#!/bin/bash

# Script to deploy Lambda layers for the SheetAssist API and update the function configuration

# Exit on error
set -e

# Get the stage from the command line or default to prod
STAGE=${1:-prod}

# Function to create a layer and return its ARN
function create_layer() {
    layer_name=$1
    req_file=$2
    
    echo "Creating layer: $layer_name from $req_file"
    
    # Create temp directory
    mkdir -p "/tmp/$layer_name/python"
    
    # Install dependencies
    pip install -r "$req_file" -t "/tmp/$layer_name/python"
    
    # Create zip file
    cd "/tmp/$layer_name"
    zip -r "/tmp/$layer_name.zip" .
    cd -
    
    # Publish layer
    LAYER_ARN=$(aws lambda publish-layer-version \
        --layer-name "${STAGE}-$layer_name" \
        --description "Layer for $layer_name in $STAGE environment" \
        --zip-file "fileb:///tmp/$layer_name.zip" \
        --compatible-runtimes python3.11 \
        --query "LayerVersionArn" \
        --output text)
        
    # Clean up
    rm -rf "/tmp/$layer_name" "/tmp/$layer_name.zip"
    
    echo $LAYER_ARN
}

echo "Creating Lambda layers for ${STAGE} environment..."

# Create all layers and capture their ARNs
CORE_LAYER_ARN=$(create_layer "sheet-assist-core" "layer-core.txt")
DATA_LAYER_ARN=$(create_layer "sheet-assist-data" "layer-data.txt")
LLM_LAYER_ARN=$(create_layer "sheet-assist-llm" "layer-llm.txt")
GOOGLE_LAYER_ARN=$(create_layer "sheet-assist-google" "layer-google.txt")
UTILS_LAYER_ARN=$(create_layer "sheet-assist-utils" "layer-utils.txt")

echo "Layer ARNs:"
echo "Core: $CORE_LAYER_ARN"
echo "Data: $DATA_LAYER_ARN"
echo "LLM: $LLM_LAYER_ARN"
echo "Google: $GOOGLE_LAYER_ARN"
echo "Utils: $UTILS_LAYER_ARN"

# Get the Lambda function name
FUNCTION_NAME="${STAGE}-sheet-assist-api-api"

echo "Updating Lambda function: $FUNCTION_NAME to use the new layers..."

# Update the Lambda function configuration to use the new layers
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --layers $CORE_LAYER_ARN $DATA_LAYER_ARN $LLM_LAYER_ARN $GOOGLE_LAYER_ARN $UTILS_LAYER_ARN

echo "Lambda function updated successfully!" 