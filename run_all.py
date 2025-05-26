import os
import sys
import subprocess
import logging
import time
from pathlib import Path
import threading
import queue
import psutil
import signal
from logging_config import health_logger, catdog_logger, imdb_logger, master_logger, project_status
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/master.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('master')

class ProjectRunner:
    def __init__(self):
        self.processes = {}
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()

    def run_health_model(self):
        """Run the Health Model project"""
        try:
            health_logger.info("Starting Health Model project...")
            project_status.update_status('health_model', 'starting')
            
            # Run the health model
            process = subprocess.Popen([sys.executable, 'run_health_app.py'],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
            
            self.processes['health'] = process
            self.monitor_process('health', process)
            
            # Monitor the process
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    health_logger.info(output.strip())
                error = process.stderr.readline()
                if error:
                    health_logger.error(error.strip())
                    project_status.update_status('health_model', 'error', error=error.strip())
            
            if process.returncode == 0:
                health_logger.info("Health Model project completed successfully")
                project_status.update_status('health_model', 'completed')
            else:
                health_logger.error(f"Health Model project failed with return code {process.returncode}")
                project_status.update_status('health_model', 'failed', 
                                          error=f"Process failed with return code {process.returncode}")
        
        except Exception as e:
            health_logger.error(f"Error in Health Model project: {str(e)}")
            project_status.update_status('health_model', 'error', error=str(e))

    def run_cat_dog_classifier(self):
        """Run the Cat and Dog Classifier project"""
        try:
            catdog_logger.info("Starting Cat and Dog Classifier project...")
            project_status.update_status('catdog_classifier', 'starting')
            
            # Run the classifier
            process = subprocess.Popen([sys.executable, 'run_simple.py'],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
            
            self.processes['cat_dog'] = process
            self.monitor_process('cat_dog', process)
            
            # Monitor the process
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    catdog_logger.info(output.strip())
                error = process.stderr.readline()
                if error:
                    catdog_logger.error(error.strip())
                    project_status.update_status('catdog_classifier', 'error', error=error.strip())
            
            if process.returncode == 0:
                catdog_logger.info("Cat and Dog Classifier project completed successfully")
                project_status.update_status('catdog_classifier', 'completed')
            else:
                catdog_logger.error(f"Cat and Dog Classifier project failed with return code {process.returncode}")
                project_status.update_status('catdog_classifier', 'failed', 
                                          error=f"Process failed with return code {process.returncode}")
        
        except Exception as e:
            catdog_logger.error(f"Error in Cat and Dog Classifier project: {str(e)}")
            project_status.update_status('catdog_classifier', 'error', error=str(e))

    def run_imdb_sentiment(self):
        """Run the IMDB Sentiment Analysis project"""
        try:
            imdb_logger.info("Starting IMDB Sentiment Analysis project...")
            project_status.update_status('imdb_sentiment', 'starting')
            
            # Run the IMDB sentiment analysis
            process = subprocess.Popen([sys.executable, 'imdb_sentiment/run.py', '--download', '--train'],
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     universal_newlines=True)
            
            self.processes['imdb'] = process
            self.monitor_process('imdb', process)
            
            # Monitor the process
            while process.poll() is None:
                output = process.stdout.readline()
                if output:
                    imdb_logger.info(output.strip())
                error = process.stderr.readline()
                if error:
                    imdb_logger.error(error.strip())
                    project_status.update_status('imdb_sentiment', 'error', error=error.strip())
            
            if process.returncode == 0:
                imdb_logger.info("IMDB Sentiment Analysis project completed successfully")
                project_status.update_status('imdb_sentiment', 'completed')
            else:
                imdb_logger.error(f"IMDB Sentiment Analysis project failed with return code {process.returncode}")
                project_status.update_status('imdb_sentiment', 'failed', 
                                          error=f"Process failed with return code {process.returncode}")
        
        except Exception as e:
            imdb_logger.error(f"Error in IMDB Sentiment Analysis project: {str(e)}")
            project_status.update_status('imdb_sentiment', 'error', error=str(e))

    def monitor_process(self, name, process):
        """Monitor a process and collect its output"""
        def monitor():
            while not self.stop_event.is_set():
                output = process.stdout.readline()
                if output:
                    self.output_queue.put(f"[{name}] {output.strip()}")
                
                error = process.stderr.readline()
                if error:
                    self.output_queue.put(f"[{name}] ERROR: {error.strip()}")
                
                if process.poll() is not None:
                    break
                
                time.sleep(0.1)
        
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()

    def print_output(self):
        """Print output from all processes"""
        while not self.stop_event.is_set():
            try:
                output = self.output_queue.get(timeout=1)
                print(output)
            except queue.Empty:
                continue

    def cleanup(self):
        """Clean up all running processes"""
        logging.info("Cleaning up processes...")
        self.stop_event.set()
        
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # If process is still running
                    process.terminate()
                    process.wait(timeout=5)
            except Exception as e:
                logging.error(f"Error stopping {name} process: {e}")
                
                # Force kill if necessary
                try:
                    parent = psutil.Process(process.pid)
                    children = parent.children(recursive=True)
                    for child in children:
                        child.kill()
                    parent.kill()
                except:
                    pass

def monitor_status():
    while True:
        master_logger.info("\nCurrent Project Status:")
        for project, status in project_status.get_status().items():
            master_logger.info(f"{project}: {status['status']}")
            if status['errors']:
                master_logger.error(f"Errors in {project}:")
                for error in status['errors']:
                    master_logger.error(f"  - {error}")
        time.sleep(30)  # Update status every 30 seconds

def run_service(script_name, port):
    """Run a service and return its process."""
    try:
        process = subprocess.Popen(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"{script_name}: started")
        return process
    except Exception as e:
        logger.error(f"Error starting {script_name}: {str(e)}")
        return None

def update_status(service_name, status, error=None):
    """Update the status of a service in the status file."""
    status_file = Path('logs/status.json')
    if status_file.exists():
        with open(status_file, 'r') as f:
            status_data = json.load(f)
    else:
        status_data = {}
    
    status_data[service_name] = {
        'status': status,
        'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
        'error': str(error) if error else None
    }
    
    with open(status_file, 'w') as f:
        json.dump(status_data, f, indent=2)

def main():
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Start services
    services = {
        'health_model': ('run_health_app.py', 8501),
        'catdog_classifier': ('catdog_classifier/run.py', 7860),
        'imdb_sentiment': ('imdb_sentiment/run.py', None)
    }
    
    processes = {}
    
    try:
        for service_name, (script, port) in services.items():
            process = run_service(script, port)
            if process:
                processes[service_name] = process
                update_status(service_name, 'running')
            else:
                update_status(service_name, 'failed', 'Failed to start')
        
        # Monitor processes
        while True:
            for service_name, process in list(processes.items()):
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    if process.returncode == 0:
                        logger.info(f"{service_name}: completed successfully")
                        update_status(service_name, 'completed')
                    else:
                        logger.error(f"Errors in {service_name}:")
                        logger.error(f"  - {stderr}")
                        update_status(service_name, 'failed', stderr)
                    del processes[service_name]
            
            if not processes:
                logger.info("All services have completed.")
                break
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down services...")
        for process in processes.values():
            process.terminate()
        logger.info("All services terminated.")

if __name__ == "__main__":
    main() 