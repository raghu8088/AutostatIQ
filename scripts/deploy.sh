#!/bin/bash

# AutoStatIQ Deployment Script
set -e

echo "🚀 AutoStatIQ Deployment Script"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "📥 Download from: https://www.docker.com/get-started"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  IMPORTANT: Please update the .env file with your OpenAI API key!"
    echo "   Edit .env and add: OPENAI_API_KEY=your_api_key_here"
    read -p "Press Enter after updating the .env file..."
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p uploads results static/plots logs

# Pull latest images if specified
if [[ "$1" == "--pull" ]]; then
    echo "📥 Pulling latest Docker images..."
    docker-compose pull
fi

# Build images
echo "🏗️  Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting AutoStatIQ services..."
docker-compose up -d

# Wait for application to start
echo "⏳ Waiting for application to start..."
sleep 15

# Health check
echo "🏥 Performing health check..."
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        echo "✅ AutoStatIQ is running successfully!"
        echo ""
        echo "🌐 Application URLs:"
        echo "   Main App:     http://localhost:5000"
        echo "   API Docs:     http://localhost:5000/api-docs"
        echo "   Health Check: http://localhost:5000/health"
        echo ""
        echo "📊 AutoStatIQ is ready for statistical analysis!"
        exit 0
    else
        echo "⏳ Attempt $((RETRY_COUNT + 1))/$MAX_RETRIES failed, retrying in 5 seconds..."
        sleep 5
        RETRY_COUNT=$((RETRY_COUNT + 1))
    fi
done

echo "❌ Health check failed after $MAX_RETRIES attempts."
echo "📋 Checking logs for issues..."
docker-compose logs --tail=50 autostatiq

echo ""
echo "🔧 Troubleshooting steps:"
echo "1. Check if port 5000 is available: lsof -i :5000"
echo "2. Verify .env file has correct OpenAI API key"
echo "3. Check Docker logs: docker-compose logs autostatiq"
echo "4. Restart services: docker-compose restart"

exit 1
