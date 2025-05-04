#!/bin/bash

# Script to deploy Lambda layers for the SheetAssist API and update the existing function

# Exit on error
set -e

STAGE="prod"
FUNCTION_NAME="sheet-assist-api-prod-api"

echo "Deploying Lambda layers for ${STAGE} environment..."
echo "Target function: ${FUNCTION_NAME}"

# Function to create a layer and return its ARN
function create_layer() {
    layer_name=$1
    req_file=$2
    
    echo "Creating layer: $layer_name from $req_file"
    
    # Create temp directory
    mkdir -p "/tmp/$layer_name/python"
    
    # Install dependencies with --ignore-installed flag to avoid warnings
    pip install --ignore-installed -r "$req_file" -t "/tmp/$layer_name/python"
    
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

echo "Updating Lambda function configuration..."

# Create a temporary JSON file for the Lambda update
TEMP_CLI_INPUT="/tmp/lambda_layers_input.json"
cat > $TEMP_CLI_INPUT << EOL
{
  "FunctionName": "$FUNCTION_NAME",
  "Layers": [
    "$CORE_LAYER_ARN",
    "$DATA_LAYER_ARN",
    "$LLM_LAYER_ARN",
    "$GOOGLE_LAYER_ARN",
    "$UTILS_LAYER_ARN"
  ]
}
EOL

# Update the Lambda function using the CLI input file
aws lambda update-function-configuration --cli-input-json file://$TEMP_CLI_INPUT

# Clean up temp file
rm $TEMP_CLI_INPUT

echo "Lambda function updated successfully!"

# Create a simplified requirements.txt for the function package
cat > requirements-lambda.txt << EOL
uvicorn==0.32.0
importlib-metadata==8.5.0
itsdangerous==2.2.0
jinja2==3.1.4
markupsafe==3.0.2
orjson==3.10.11
ujson==5.10.0
python-dateutil==2.9.0.post0
EOL

echo "Created simplified requirements-lambda.txt file."
echo "To deploy the updated function with the minimal requirements, run:"
echo "serverless deploy --stage ${STAGE} --function api"
echo ""
echo "Lambda layers implementation complete!" 