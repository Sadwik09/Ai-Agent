import os
import sys
import subprocess
import time
import signal
import psutil
import logging
import asyncio
import nest_asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('health_app.log'),
        logging.StreamHandler()
    ]
)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'torch', 'numpy', 'pandas', 'scikit-learn', 'matplotlib',
        'streamlit', 'openai', 'google-generativeai', 'plotly',
        'python-dotenv', 'seaborn', 'streamlit-plotly-events',
        'streamlit-autorefresh', 'psutil', 'asyncio', 'nest-asyncio'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logging.warning(f"Missing packages: {', '.join(missing_packages)}")
        logging.info("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
        logging.info("All required packages installed successfully.")

def check_environment():
    """Check and create necessary environment files and directories."""
    # Check .env file
    env_path = Path('health_model/.env')
    if not env_path.exists():
        logging.info("Creating .env file...")
        with open(env_path, 'w') as f:
            f.write("""# API Keys
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
""")
        logging.info("Please update the .env file with your API keys.")
    
    # Check model directory
    model_dir = Path('health_model/models')
    if not model_dir.exists():
        logging.info("Creating model directory...")
        model_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if model file exists
    model_path = model_dir / 'best_model.pth'
    if not model_path.exists():
        logging.warning("Model file not found. Using default model.")

def kill_existing_streamlit():
    """Kill any existing Streamlit processes."""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'streamlit' in ' '.join(proc.info['cmdline'] or []):
                logging.info(f"Killing existing Streamlit process (PID: {proc.info['pid']})")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

def run_application():
    """Run the Streamlit application with proper error handling."""
    try:
        # Kill any existing Streamlit processes
        kill_existing_streamlit()
        
        # Check dependencies
        check_dependencies()
        
        # Check environment
        check_environment()
        
        # Get the absolute path to the app
        app_path = Path('health_model/src/app.py').absolute()
        
        # Set environment variables for Streamlit
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path.cwd())
        
        # Run Streamlit with event loop handling
        logging.info("Starting Streamlit application...")
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env
        )
        
        # Monitor the process
        while True:
            output = process.stdout.readline()
            if output:
                logging.info(output.strip())
            
            error = process.stderr.readline()
            if error:
                logging.error(error.strip())
            
            # Check if process is still running
            if process.poll() is not None:
                break
            
            time.sleep(0.1)
        
        # Get return code
        return_code = process.poll()
        if return_code != 0:
            logging.error(f"Application exited with code {return_code}")
            return False
        
        return True
    
    except KeyboardInterrupt:
        logging.info("Application stopped by user.")
        return True
    except Exception as e:
        logging.error(f"Error running application: {str(e)}")
        return False
    finally:
        # Cleanup
        kill_existing_streamlit()

def main():
    """Main function to run the application."""
    logging.info("Starting Health Prediction Application...")
    
    if run_application():
        logging.info("Application completed successfully.")
    else:
        logging.error("Application failed to run properly.")
        sys.exit(1)

if __name__ == "__main__":
    main() 