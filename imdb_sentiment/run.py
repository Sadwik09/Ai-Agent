import os
import argparse
import subprocess
import sys
import time
import shutil
from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/imdb_sentiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('imdb_sentiment')

def create_venv():
    """Creates and activates a virtual environment."""
    venv_dir = 'venv'
    venv_path = Path(venv_dir)
    
    # Remove existing venv if it exists
    if venv_path.exists():
        print(f"Removing existing virtual environment at {venv_dir}...")
        try:
            shutil.rmtree(venv_path)
            print("Existing virtual environment removed.")
        except Exception as e:
            print(f"Warning: Could not remove existing venv: {e}")
            print("Continuing with existing venv...")

    # Create new venv
    print(f"Creating virtual environment at {venv_dir}...")
    try:
        subprocess.run([sys.executable, '-m', 'venv', venv_dir], check=True)
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

    # Get the path to the Python executable in the venv
    if sys.platform == 'win32':
        venv_python = venv_path / 'Scripts' / 'python.exe'
    else:
        venv_python = venv_path / 'bin' / 'python'

    # Wait for the file to be created
    max_attempts = 5
    for attempt in range(max_attempts):
        if venv_python.exists():
            break
        print(f"Waiting for Python executable to be created (attempt {attempt + 1}/{max_attempts})...")
        time.sleep(2)
    else:
        print(f"Error: Could not find Python executable at {venv_python}")
        print("Please try running the script again.")
        sys.exit(1)

    print(f"Using Python interpreter: {venv_python}")
    return str(venv_python)

def install_requirements(python_executable):
    """Installs dependencies from requirements.txt using the specified python executable."""
    requirements_file = 'requirements.txt'
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)
        
    print(f"Installing dependencies from {requirements_file}...")
    try:
        # Install requirements directly
        subprocess.run([python_executable, '-m', 'pip', 'install', '-r', requirements_file], check=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def setup_environment():
    """Set up the environment for IMDB sentiment analysis."""
    # Add the project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    
    # Create necessary directories
    data_dir = project_root / "data" / "imdb"
    models_dir = project_root / "models" / "imdb"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load and preprocess the IMDB dataset."""
    logger.info("Loading IMDB dataset...")
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    # Pad sequences
    x_train = pad_sequences(x_train, maxlen=200)
    x_test = pad_sequences(x_test, maxlen=200)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    """Create and compile the model."""
    logger.info("Creating model...")
    model = Sequential([
        Embedding(10000, 16, input_length=200),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, x_train, y_train, x_test, y_test):
    """Train the model."""
    logger.info("Training model...")
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=512,
        validation_data=(x_test, y_test),
        verbose=1
    )
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    model.save('models/imdb_model.h5')
    logger.info("Model saved to models/imdb_model.h5")
    
    return history

def main():
    try:
        # Load data
        (x_train, y_train), (x_test, y_test) = load_data()
        
        # Create and train model
        model = create_model()
        history = train_model(model, x_train, y_train, x_test, y_test)
        
        # Evaluate model
        test_loss, test_acc = model.evaluate(x_test, y_test)
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        logger.info("IMDB sentiment analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in IMDB sentiment analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 