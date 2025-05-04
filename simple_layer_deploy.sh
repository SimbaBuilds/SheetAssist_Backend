#!/bin/bash

# Simple script to deploy Lambda layers for SheetAssist API
set -e

FUNCTION_NAME="sheet-assist-api-prod-api"

# Create layers from existing layer requirement files
echo "Creating and uploading layers..."

# Create core layer
echo "Creating core layer..."
mkdir -p /tmp/layer-core/python
pip install -r layer-core.txt -t /tmp/layer-core/python
cd /tmp/layer-core && zip -r ../layer-core.zip .
cd -
CORE_LAYER=$(aws lambda publish-layer-version \
  --layer-name prod-sheet-assist-core \
  --description "Core dependencies" \
  --zip-file fileb:///tmp/layer-core.zip \
  --compatible-runtimes python3.11 \
  --query 'LayerVersionArn' \
  --output text)
echo "Core layer ARN: $CORE_LAYER"

# Create data layer
echo "Creating data layer..."
mkdir -p /tmp/layer-data/python
pip install -r layer-data.txt -t /tmp/layer-data/python
cd /tmp/layer-data && zip -r ../layer-data.zip .
cd -
DATA_LAYER=$(aws lambda publish-layer-version \
  --layer-name prod-sheet-assist-data \
  --description "Data dependencies" \
  --zip-file fileb:///tmp/layer-data.zip \
  --compatible-runtimes python3.11 \
  --query 'LayerVersionArn' \
  --output text)
echo "Data layer ARN: $DATA_LAYER"

# Create LLM layer
echo "Creating LLM layer..."
mkdir -p /tmp/layer-llm/python
pip install -r layer-llm.txt -t /tmp/layer-llm/python
cd /tmp/layer-llm && zip -r ../layer-llm.zip .
cd -
LLM_LAYER=$(aws lambda publish-layer-version \
  --layer-name prod-sheet-assist-llm \
  --description "LLM dependencies" \
  --zip-file fileb:///tmp/layer-llm.zip \
  --compatible-runtimes python3.11 \
  --query 'LayerVersionArn' \
  --output text)
echo "LLM layer ARN: $LLM_LAYER"

# Create Google layer
echo "Creating Google layer..."
mkdir -p /tmp/layer-google/python
pip install -r layer-google.txt -t /tmp/layer-google/python
cd /tmp/layer-google && zip -r ../layer-google.zip .
cd -
GOOGLE_LAYER=$(aws lambda publish-layer-version \
  --layer-name prod-sheet-assist-google \
  --description "Google dependencies" \
  --zip-file fileb:///tmp/layer-google.zip \
  --compatible-runtimes python3.11 \
  --query 'LayerVersionArn' \
  --output text)
echo "Google layer ARN: $GOOGLE_LAYER"

# Create Utils layer
echo "Creating Utils layer..."
mkdir -p /tmp/layer-utils/python
pip install -r layer-utils.txt -t /tmp/layer-utils/python
cd /tmp/layer-utils && zip -r ../layer-utils.zip .
cd -
UTILS_LAYER=$(aws lambda publish-layer-version \
  --layer-name prod-sheet-assist-utils \
  --description "Utils dependencies" \
  --zip-file fileb:///tmp/layer-utils.zip \
  --compatible-runtimes python3.11 \
  --query 'LayerVersionArn' \
  --output text)
echo "Utils layer ARN: $UTILS_LAYER"

# Update Lambda function with layers
echo "Updating Lambda function with layers..."

# Update Lambda function directly with ARNs
aws lambda update-function-configuration \
  --function-name $FUNCTION_NAME \
  --layers "$CORE_LAYER" "$DATA_LAYER" "$LLM_LAYER" "$GOOGLE_LAYER" "$UTILS_LAYER"

echo "Lambda layers deployed successfully!"
echo "You can now deploy the function with minimal requirements using:"
echo "serverless deploy --stage prod --function api" 