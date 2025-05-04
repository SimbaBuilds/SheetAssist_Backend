#!/bin/bash

STAGE=$1
if [ -z "$STAGE" ]; then
  STAGE="dev"
fi

echo "Deploying Minimal SheetAssist API to $STAGE environment..."
echo "Deploying with Serverless Framework..."
npx serverless deploy --config serverless-minimal.yml --stage $STAGE 