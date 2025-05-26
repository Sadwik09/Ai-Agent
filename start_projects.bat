@echo off
echo Setting up AI Agent Projects...

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
pip install streamlit plotly pandas nest-asyncio streamlit-plotly-events streamlit-autorefresh

:: Create necessary directories
echo Creating directories...
mkdir data 2>nul
mkdir models 2>nul
mkdir results 2>nul
mkdir logs 2>nul

:: Run the application
echo Starting all projects...
python run_all.py

pause 