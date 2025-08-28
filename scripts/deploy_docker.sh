#!/bin/bash

# MindModel Docker Deployment Script
# This script deploys the MindModel service to a Docker host

set -e

echo "🚀 Deploying MindModel to Docker host..."

# Configuration
CONTAINER_NAME="mindmodel"
IMAGE_NAME="mindmodel:latest"
PORT="8000"
DATA_DIR="/opt/mindmodel"

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

# Stop and remove existing container
echo "🛑 Stopping existing container..."
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Create necessary directories
echo "📁 Creating data directories..."
sudo mkdir -p $DATA_DIR/data
sudo mkdir -p $DATA_DIR/models
sudo mkdir -p $DATA_DIR/runs
sudo chown -R $USER:$USER $DATA_DIR

# Run the new container
echo "🚀 Starting new container..."
docker run -d \
  --name $CONTAINER_NAME \
  --restart unless-stopped \
  -p $PORT:8000 \
  -v $DATA_DIR/data:/app/data \
  -v $DATA_DIR/models:/app/models \
  -v $DATA_DIR/runs:/app/runs \
  -e MINDMODEL_MODEL=google/flan-t5-base \
  $IMAGE_NAME

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
for i in {1..60}; do
  if curl -f http://localhost:$PORT/health 2>/dev/null; then
    echo "✅ Service is healthy!"
    break
  fi
  if [ $i -eq 60 ]; then
    echo "❌ Service failed to start properly"
    docker logs $CONTAINER_NAME
    exit 1
  fi
  echo "⏳ Waiting... ($i/60)"
  sleep 5
done

# Test the deployment
echo "🧪 Testing deployment..."

# Test health endpoint
curl -f http://localhost:$PORT/health

# Test status endpoint
curl -f http://localhost:$PORT/status

# Test conclusion generation
curl -X POST http://localhost:$PORT/v1/conclude \
  -H "Content-Type: application/json" \
  -d '{"input": "The company reported $3.2M revenue in Q2 2024."}' \
  -f

echo "✅ All tests passed!"

# Show deployment info
echo "🎉 Deployment completed successfully!"
echo ""
echo "📋 Service Information:"
echo "  - Web UI: http://10.11.2.6:8000"
echo "  - API Base: http://10.11.2.6:8000"
echo "  - Health Check: http://10.11.2.6:8000/health"
echo "  - Status: http://10.11.2.6:8000/status"
echo ""
echo "📋 Web UI Features:"
echo "  - Test: Generate conclusions interactively"
echo "  - Teach: Add training examples one by one"
echo "  - Batch Upload: Upload JSONL files for training"
echo "  - Status: Monitor system and model status"
echo ""
echo "📋 API Endpoints:"
echo "  - POST /v1/conclude - Generate conclusions"
echo "  - POST /v1/learn - Train the model"
echo "  - GET /v1/models - List trained models"
echo ""
echo "📋 Container status:"
docker ps | grep $CONTAINER_NAME
echo ""
echo "📋 Recent logs:"
docker logs --tail 10 $CONTAINER_NAME
