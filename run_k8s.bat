@echo off
echo Setting up AI Agent Projects with Kubernetes...

:: Set environment variables
set DOCKER_REGISTRY=your-registry
set OPENAI_API_KEY=your-openai-key
set GEMINI_API_KEY=your-gemini-key
set KAGGLE_USERNAME=your-kaggle-username
set KAGGLE_KEY=your-kaggle-key

:: Create and activate virtual environment
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
echo Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install kubernetes docker

:: Create necessary directories
echo Creating directories...
mkdir data 2>nul
mkdir models 2>nul
mkdir results 2>nul
mkdir logs 2>nul
mkdir k8s 2>nul

:: Build and deploy to Kubernetes
echo Building and deploying to Kubernetes...
docker build -t %DOCKER_REGISTRY%/ai-agent:latest .
docker push %DOCKER_REGISTRY%/ai-agent:latest

:: Apply Kubernetes configurations
echo Applying Kubernetes configurations...
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

:: Wait for deployment
echo Waiting for deployment to be ready...
kubectl rollout status deployment/ai-agent

:: Get service URLs
echo Getting service URLs...
for /f "tokens=*" %%a in ('kubectl get service ai-agent-service -o jsonpath="{.status.loadBalancer.ingress[0].ip}:8501"') do set STREAMLIT_URL=%%a
for /f "tokens=*" %%a in ('kubectl get service ai-agent-service -o jsonpath="{.status.loadBalancer.ingress[0].ip}:7860"') do set GRADIO_URL=%%a

echo Deployment complete!
echo Streamlit URL: http://%STREAMLIT_URL%
echo Gradio URL: http://%GRADIO_URL%

pause 