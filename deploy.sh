#!/bin/bash

# Exit on error
set -e

# Load environment variables
if [ -f .env.production ]; then
    source .env.production
fi

# Pull latest changes
git pull origin main

# Build and start containers
docker-compose -f docker-compose.yml build
docker-compose -f docker-compose.yml up -d

# Run database migrations (if needed)
# docker-compose exec ai-agent python manage.py migrate

# Collect static files (if needed)
# docker-compose exec ai-agent python manage.py collectstatic --noinput

# Check if services are healthy
echo "Checking service health..."
sleep 10
docker-compose ps

# Monitor logs
echo "Monitoring logs..."
docker-compose logs -f 