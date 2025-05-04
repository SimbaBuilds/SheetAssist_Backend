#!/bin/bash

# Script to deploy the SheetAssist API to AWS Lambda

# Exit on error
set -e

# Get the stage from the command line or default to dev
STAGE=${1:-dev}

# Check if the --remove flag is provided
REMOVE=false
if [ "$2" == "--remove" ]; then
    REMOVE=true
fi

# Check if the --skip-layers flag is provided
SKIP_LAYERS=false
if [ "$2" == "--skip-layers" ] || [ "$3" == "--skip-layers" ]; then
    SKIP_LAYERS=true
fi

echo "Deploying SheetAssist API to ${STAGE} environment..."

# Check if env file exists
if [ ! -f "env-${STAGE}.yml" ]; then
    echo "Error: env-${STAGE}.yml file not found!"
    echo "Please create it by copying env-example.yml and updating the values."
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing npm dependencies..."
    npm install
fi

# Remove existing deployment if requested
if [ "$REMOVE" = true ]; then
    echo "Removing existing deployment..."
    npx serverless remove --stage ${STAGE} || true
    echo "Waiting for resources to be fully removed..."
    sleep 10
fi

# Create and publish Lambda layers if not skipped
if [ "$SKIP_LAYERS" = false ]; then
    echo "Creating Lambda layers..."
    chmod +x ./create_layers.sh
    
    # Create layers and capture the ARNs
    CORE_LAYER_ARN=$(./create_layers.sh ${STAGE} | grep "sheet-assist-core" | tail -1)
    DATA_LAYER_ARN=$(./create_layers.sh ${STAGE} | grep "sheet-assist-data" | tail -1)
    LLM_LAYER_ARN=$(./create_layers.sh ${STAGE} | grep "sheet-assist-llm" | tail -1)
    GOOGLE_LAYER_ARN=$(./create_layers.sh ${STAGE} | grep "sheet-assist-google" | tail -1)
    UTILS_LAYER_ARN=$(./create_layers.sh ${STAGE} | grep "sheet-assist-utils" | tail -1)
    
    echo "Layer ARNs:"
    echo "Core: $CORE_LAYER_ARN"
    echo "Data: $DATA_LAYER_ARN"
    echo "LLM: $LLM_LAYER_ARN"
    echo "Google: $GOOGLE_LAYER_ARN"
    echo "Utils: $UTILS_LAYER_ARN"
else
    echo "Skipping Lambda layer creation..."
    # You'll need to provide the ARNs manually or get them from a configuration file
    CORE_LAYER_ARN="CORE_LAYER_ARN_PLACEHOLDER"
    DATA_LAYER_ARN="DATA_LAYER_ARN_PLACEHOLDER"
    LLM_LAYER_ARN="LLM_LAYER_ARN_PLACEHOLDER"
    GOOGLE_LAYER_ARN="GOOGLE_LAYER_ARN_PLACEHOLDER"
    UTILS_LAYER_ARN="UTILS_LAYER_ARN_PLACEHOLDER"
fi

# Deploy the service with layer ARNs
echo "Deploying with Serverless Framework..."
npx serverless deploy --stage ${STAGE} --verbose \
    --param="coreLayerArn=${CORE_LAYER_ARN}" \
    --param="dataLayerArn=${DATA_LAYER_ARN}" \
    --param="llmLayerArn=${LLM_LAYER_ARN}" \
    --param="googleLayerArn=${GOOGLE_LAYER_ARN}" \
    --param="utilsLayerArn=${UTILS_LAYER_ARN}"

# Display information about the deployed service
echo "Fetching information about the deployed service..."
npx serverless info --stage ${STAGE}

echo "Deployment complete!" 