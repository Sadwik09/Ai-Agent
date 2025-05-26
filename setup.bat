@echo off
echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

echo Setup complete! You can now run the applications.
echo To run all projects: python run_all.py
echo To run individual projects:
echo - Health Model: python run_health_app.py
echo - Cat and Dog Classifier: python run_simple.py
echo - IMDB Sentiment: python imdb_sentiment/run.py --download --train

pause 