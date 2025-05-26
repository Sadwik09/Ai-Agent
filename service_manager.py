import os
import sys
import subprocess
import time
import logging
import signal
import psutil
from pathlib import Path
from typing import Dict, Optional
from project_config import config

class ServiceManager:
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.logger = self._setup_logger()
        self.stop_event = False
    
    def _setup_logger(self) -> logging.Logger:
        """Set up the logger for the service manager."""
        logger = logging.getLogger('service_manager')
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(config.get_log_path('service_manager'))
        file_handler.setFormatter(logging.Formatter(config.logging['format']))
        logger.addHandler(file_handler)
        
        # Console handler
        if config.logging['console']['enabled']:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(config.logging['format']))
            logger.addHandler(console_handler)
        
        return logger
    
    def start_service(self, service_name: str) -> bool:
        """Start a specific service."""
        service_config = config.get_service_config(service_name)
        if not service_config:
            self.logger.error(f"Service {service_name} not found in configuration")
            return False
        
        try:
            # Check if service is already running
            if service_name in self.processes and self.processes[service_name].poll() is None:
                self.logger.info(f"Service {service_name} is already running")
                return True
            
            # Start the service
            self.logger.info(f"Starting {service_config['name']}...")
            
            if service_config['type'] == 'streamlit':
                process = subprocess.Popen(
                    [sys.executable, '-m', 'streamlit', 'run', service_config['script']],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, service_config['script']],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            self.processes[service_name] = process
            self.logger.info(f"{service_config['name']} started successfully")
            
            # Start monitoring thread
            self._monitor_process(service_name, process)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting {service_name}: {str(e)}")
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a specific service."""
        if service_name not in self.processes:
            self.logger.warning(f"Service {service_name} is not running")
            return True
        
        try:
            process = self.processes[service_name]
            if process.poll() is None:  # Process is still running
                self.logger.info(f"Stopping {service_name}...")
                
                # Try graceful termination first
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    process.kill()
                    process.wait()
                
                self.logger.info(f"{service_name} stopped successfully")
            
            del self.processes[service_name]
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping {service_name}: {str(e)}")
            return False
    
    def _monitor_process(self, service_name: str, process: subprocess.Popen):
        """Monitor a process and log its output."""
        def monitor():
            while not self.stop_event:
                # Read stdout
                output = process.stdout.readline()
                if output:
                    self.logger.info(f"[{service_name}] {output.strip()}")
                
                # Read stderr
                error = process.stderr.readline()
                if error:
                    self.logger.error(f"[{service_name}] {error.strip()}")
                
                # Check if process has ended
                if process.poll() is not None:
                    self.logger.info(f"{service_name} process ended with code {process.returncode}")
                    break
                
                time.sleep(0.1)
        
        import threading
        thread = threading.Thread(target=monitor)
        thread.daemon = True
        thread.start()
    
    def start_all_services(self):
        """Start all configured services."""
        self.logger.info("Starting all services...")
        for service_name in config.services:
            self.start_service(service_name)
    
    def stop_all_services(self):
        """Stop all running services."""
        self.logger.info("Stopping all services...")
        self.stop_event = True
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
    
    def get_service_status(self, service_name: str) -> dict:
        """Get the status of a specific service."""
        service_config = config.get_service_config(service_name)
        if not service_config:
            return {'status': 'not_found'}
        
        if service_name not in self.processes:
            return {'status': 'stopped'}
        
        process = self.processes[service_name]
        if process.poll() is None:
            return {
                'status': 'running',
                'pid': process.pid,
                'url': service_config['url']
            }
        else:
            return {
                'status': 'stopped',
                'exit_code': process.returncode
            }
    
    def get_all_status(self) -> dict:
        """Get the status of all services."""
        return {
            service_name: self.get_service_status(service_name)
            for service_name in config.services
        }

def main():
    manager = ServiceManager()
    
    try:
        # Start all services
        manager.start_all_services()
        
        # Print status
        print("\nService Status:")
        for service_name, status in manager.get_all_status().items():
            print(f"{service_name}: {status['status']}")
            if status['status'] == 'running' and 'url' in status:
                print(f"URL: {status['url']}")
        
        print("\nPress Ctrl+C to stop all services...")
        
        # Keep the script running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping all services...")
        manager.stop_all_services()
        print("All services stopped.")

if __name__ == "__main__":
    main() 