import os
import sys
import subprocess
import time
from pathlib import Path

def run_command(command):
    """Run a command and return its process."""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except Exception as e:
        print(f"Error running command '{command}': {str(e)}")
        return None

def main():
    # Create necessary directories
    for directory in ['data', 'models', 'results', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.run([sys.executable, "-m", "pip", "install", 
                   "scikit-learn", "google-generativeai", "python-dotenv",
                   "streamlit-autorefresh", "nest-asyncio", "streamlit",
                   "plotly", "pandas", "tensorflow", "gradio"])
    
    # Start services
    print("\nStarting services...")
    
    # Start Streamlit app
    streamlit_process = run_command("streamlit run run_health_app.py")
    if streamlit_process:
        print("Streamlit app started at http://localhost:8501")
    
    # Start Gradio app
    gradio_process = run_command("python catdog_classifier/run.py")
    if gradio_process:
        print("Gradio app started at http://localhost:7860")
    
    # Start IMDB sentiment analysis
    imdb_process = run_command("python imdb_sentiment/run.py")
    if imdb_process:
        print("IMDB sentiment analysis started")
    
    print("\nAll services started!")
    print("Press Ctrl+C to stop all services...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        for process in [streamlit_process, gradio_process, imdb_process]:
            if process:
                process.terminate()
        print("All services stopped.")

if __name__ == "__main__":
    main()
