#!/bin/bash

# Exit on error
set -e

# Get the stage from the command line or default to dev
STAGE=${1:-dev}

echo "Creating Lambda layers for ${STAGE} environment..."

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
    aws lambda publish-layer-version \
        --layer-name "${STAGE}-$layer_name" \
        --description "Layer for $layer_name in $STAGE environment" \
        --zip-file "fileb:///tmp/$layer_name.zip" \
        --compatible-runtimes python3.11
        
    # Clean up
    rm -rf "/tmp/$layer_name" "/tmp/$layer_name.zip"
    
    # Output the ARN of the latest version
    aws lambda list-layer-versions \
        --layer-name "${STAGE}-$layer_name" \
        --query "LayerVersions[0].LayerVersionArn" \
        --output text
}

create_layer "sheet-assist-core" "layer-core.txt"
create_layer "sheet-assist-data" "layer-data.txt"
create_layer "sheet-assist-llm" "layer-llm.txt"
create_layer "sheet-assist-google" "layer-google.txt"
create_layer "sheet-assist-utils" "layer-utils.txt"

echo "All layers have been created successfully!" 