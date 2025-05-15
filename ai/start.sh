#!/bin/bash

echo "Activating virtual environment..."
# Source the virtual environment
. /root/hellojob/venv/bin/activate

echo "Building Naver Search MCP..."
# Navigate to the MCP directory and build
cd /root/hellojob/app/mcp/naver-search-mcp
npm install

echo "Starting FastAPI application..."
# Execute the main application command
cd /root/hellojob
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload