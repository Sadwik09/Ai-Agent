@echo off
echo Setting up AI Agent Projects...

:: Run setup script
python setup.py

:: Activate virtual environment
call venv\Scripts\activate

:: Run the application
echo Starting all projects...
python run_all.py

pause 