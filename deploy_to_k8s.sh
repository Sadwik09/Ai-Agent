#!/bin/bash

# Build and push Docker image
echo "Building Docker image..."
docker build -t ${DOCKER_REGISTRY}/ai-agent:latest .

echo "Pushing Docker image..."
docker push ${DOCKER_REGISTRY}/ai-agent:latest

# Create Kubernetes secrets
echo "Creating Kubernetes secrets..."
kubectl create secret generic ai-agent-secrets \
    --from-literal=openai-api-key=${OPENAI_API_KEY} \
    --from-literal=gemini-api-key=${GEMINI_API_KEY} \
    --from-literal=kaggle-username=${KAGGLE_USERNAME} \
    --from-literal=kaggle-key=${KAGGLE_KEY} \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes configurations
echo "Applying Kubernetes configurations..."
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Wait for deployment to be ready
echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/ai-agent

# Get service URLs
echo "Getting service URLs..."
STREAMLIT_URL=$(kubectl get service ai-agent-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}:8501')
GRADIO_URL=$(kubectl get service ai-agent-service -o jsonpath='{.status.loadBalancer.ingress[0].ip}:7860')

echo "Deployment complete!"
echo "Streamlit URL: http://${STREAMLIT_URL}"
echo "Gradio URL: http://${GRADIO_URL}" 