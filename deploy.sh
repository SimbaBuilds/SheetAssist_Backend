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

# Deploy the service
echo "Deploying with Serverless Framework..."
npx serverless deploy --stage ${STAGE} --verbose

# Display information about the deployed service
echo "Fetching information about the deployed service..."
npx serverless info --stage ${STAGE}

echo "Deployment complete!" 