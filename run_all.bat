@echo off
echo Starting AI Agent Projects...

:: Activate virtual environment
call venv\Scripts\activate

:: Install dependencies
pip install -r requirements.txt
pip install scikit-learn google-generativeai python-dotenv streamlit-autorefresh nest-asyncio streamlit plotly pandas tensorflow gradio

:: Start Streamlit app
start cmd /k "streamlit run run_health_app.py"

:: Start Gradio app
start cmd /k "python catdog_classifier/run.py"

:: Start IMDB sentiment analysis
start cmd /k "python imdb_sentiment/run.py"

echo All services started!
echo Streamlit app: http://localhost:8501
echo Gradio app: http://localhost:7860

pause 