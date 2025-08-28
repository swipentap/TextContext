#!/bin/bash

set -e

echo "🚀 Deploying MindModel Learning API..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data models runs

# Build and start the service
echo "🔨 Building and starting MindModel..."
docker-compose up --build -d

# Wait for service to be ready
echo "⏳ Waiting for service to be ready..."
sleep 10

# Check health
echo "🏥 Checking service health..."
for i in {1..30}; do
    if curl -f http://localhost:8000/health &> /dev/null; then
        echo "✅ Service is healthy!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "❌ Service failed to start properly"
        docker-compose logs mindmodel
        exit 1
    fi
    echo "⏳ Waiting... ($i/30)"
    sleep 2
done

# Show status
echo "📊 Service status:"
curl -s http://localhost:8000/status | python3 -m json.tool

echo ""
echo "🎉 MindModel is now running!"
echo ""
echo "📖 Quick start:"
echo "  Generate conclusion:"
echo "    curl -X POST http://localhost:8000/v1/conclude \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"input\": \"The company reported $3.2M revenue.\"}'"
echo ""
echo "  Teach the model:"
echo "    curl -X POST http://localhost:8000/v1/learn \\"
echo "      -H 'Content-Type: application/json' \\"
echo "      -d '{\"examples\": [{\"input\": \"...\", \"target\": \"...\"}]}'"
echo ""
echo "  Check status:"
echo "    curl http://localhost:8000/status"
echo ""
echo "📝 View logs:"
echo "    docker-compose logs -f mindmodel"
echo ""
echo "🛑 Stop service:"
echo "    docker-compose down"
