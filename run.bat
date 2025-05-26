@echo off
echo Starting AI Agent Projects...

:: Check Python version
python -c "import sys; sys.exit(0 if sys.version_info >= (3,8) else 1)" || (
    echo Error: Python 3.8 or higher is required.
    echo Current version:
    python --version
    pause
    exit /b 1
)

:: Run setup
echo Running setup...
python setup.py
if errorlevel 1 (
    echo Setup failed!
    pause
    exit /b 1
)

:: Activate virtual environment and run
echo Starting services...
call venv\Scripts\activate && python run_advanced.py

pause
