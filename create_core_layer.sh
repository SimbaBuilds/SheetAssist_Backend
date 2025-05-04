#!/bin/bash

# Script to create just the core layer

# Exit on error
set -e

STAGE="prod"
LAYER_NAME="${STAGE}-sheet-assist-core"

echo "Creating layer: $LAYER_NAME"

# Create temp directory
mkdir -p "/tmp/$LAYER_NAME/python"

# Install dependencies with --no-deps to avoid dependency resolution
pip install --no-deps -r layer-core.txt -t "/tmp/$LAYER_NAME/python"

# Create zip file
cd "/tmp/$LAYER_NAME"
zip -r "/tmp/$LAYER_NAME.zip" .
cd -

# Publish layer
aws lambda publish-layer-version \
    --layer-name "$LAYER_NAME" \
    --description "Core dependencies layer for SheetAssist" \
    --zip-file "fileb:///tmp/$LAYER_NAME.zip" \
    --compatible-runtimes python3.11

# Clean up
rm -rf "/tmp/$LAYER_NAME" "/tmp/$LAYER_NAME.zip"

echo "Core layer created successfully!" 