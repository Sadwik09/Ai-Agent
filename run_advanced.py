import os
import sys
import subprocess
import time
import logging
from pathlib import Path
from project_config import config
from service_manager import ServiceManager

def setup_environment():
    """Set up the project environment."""
    print("Setting up project environment...")
    
    # Create necessary directories
    config.setup_directories()
    
    # Create and activate virtual environment
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
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([str(pip_path), 'install', '--upgrade', 'pip'], check=True)
    subprocess.run([str(pip_path), 'install', '-r', 'requirements.txt'], check=True)
    
    print("Environment setup completed!")

def main():
    """Main function to run all services."""
    try:
        # Set up environment
        setup_environment()
        
        # Initialize service manager
        manager = ServiceManager()
        
        # Start all services
        print("\nStarting all services...")
        manager.start_all_services()
        
        # Print service status
        print("\nService Status:")
        for service_name, status in manager.get_all_status().items():
            print(f"\n{service_name}:")
            print(f"  Status: {status['status']}")
            if status['status'] == 'running':
                if 'url' in status:
                    print(f"  URL: {status['url']}")
                if 'pid' in status:
                    print(f"  PID: {status['pid']}")
        
        print("\nAll services are running!")
        print("Press Ctrl+C to stop all services...")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all services...")
        manager.stop_all_services()
        print("All services stopped.")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 