import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current version: {current_version[0]}.{current_version[1]}")
        sys.exit(1)
    
    print(f"Python version {current_version[0]}.{current_version[1]} is compatible.")

def setup_environment():
    """Set up the project environment."""
    print("Setting up project environment...")
    
    # Create necessary directories
    directories = ['logs', 'models', 'data', 'test_images', 'src', 'notebooks', 'configs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    # Create virtual environment if it doesn't exist
    venv_path = Path('venv')
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
    
    # Get the path to pip in the virtual environment
    if sys.platform == 'win32':
        pip_path = venv_path / 'Scripts' / 'pip'
        python_path = venv_path / 'Scripts' / 'python'
    else:
        pip_path = venv_path / 'bin' / 'pip'
        python_path = venv_path / 'bin' / 'python'
    
    # Upgrade pip and install dependencies
    print("Installing dependencies...")
    try:
        subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], check=True)
        print("Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def main():
    """Main setup function."""
    print("Starting project setup...")
    
    # Check Python version
    check_python_version()
    
    # Set up environment
    setup_environment()
    
    print("\nSetup completed successfully!")
    print("\nTo activate the virtual environment:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    print("\nTo run the project:")
    print("    python run_advanced.py")

if __name__ == "__main__":
    main() 