import logging
import os
from pathlib import Path

# Create logs directory
Path('logs').mkdir(exist_ok=True)

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create loggers for each project
health_logger = logging.getLogger('health_model')
catdog_logger = logging.getLogger('catdog_classifier')
imdb_logger = logging.getLogger('imdb_sentiment')
master_logger = logging.getLogger('master')

# Add file handlers
for logger in [health_logger, catdog_logger, imdb_logger, master_logger]:
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'logs/{logger.name}.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

class ProjectStatus:
    def __init__(self):
        self.status = {}
    
    def update_status(self, project, status, error=None):
        if project not in self.status:
            self.status[project] = {'status': status, 'errors': []}
        else:
            self.status[project]['status'] = status
            if error:
                self.status[project]['errors'].append(error)
    
    def get_status(self):
        return self.status

project_status = ProjectStatus() 