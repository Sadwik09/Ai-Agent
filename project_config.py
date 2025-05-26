import os
from pathlib import Path

class ProjectConfig:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.directories = {
            'data': self.root_dir / 'data',
            'models': self.root_dir / 'models',
            'logs': self.root_dir / 'logs',
            'results': self.root_dir / 'results',
            'test_images': self.root_dir / 'test_images',
            'src': self.root_dir / 'src',
            'notebooks': self.root_dir / 'notebooks',
            'configs': self.root_dir / 'configs'
        }
        
        # Service configurations
        self.services = {
            'health_model': {
                'name': 'Health Model Dashboard',
                'script': 'run_health_app.py',
                'port': 8501,
                'url': 'http://localhost:8501',
                'type': 'streamlit'
            },
            'catdog_classifier': {
                'name': 'Cat and Dog Classifier',
                'script': 'catdog_classifier/run.py',
                'port': 7860,
                'url': 'http://localhost:7860',
                'type': 'gradio'
            },
            'imdb_sentiment': {
                'name': 'IMDB Sentiment Analysis',
                'script': 'imdb_sentiment/run.py',
                'port': None,
                'url': None,
                'type': 'script'
            }
        }
        
        # Model configurations
        self.models = {
            'catdog': {
                'name': 'Cat and Dog Classifier',
                'path': self.directories['models'] / 'catdog_model.h5',
                'input_shape': (224, 224, 3),
                'classes': ['cat', 'dog']
            },
            'imdb': {
                'name': 'IMDB Sentiment Analysis',
                'path': self.directories['models'] / 'imdb_model.h5',
                'vocab_size': 10000,
                'max_length': 200
            }
        }
        
        # Logging configurations
        self.logging = {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'level': 'INFO',
            'file': {
                'enabled': True,
                'path': self.directories['logs'] / 'app.log'
            },
            'console': {
                'enabled': True
            }
        }
    
    def setup_directories(self):
        """Create all necessary directories."""
        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_service_config(self, service_name):
        """Get configuration for a specific service."""
        return self.services.get(service_name)
    
    def get_model_config(self, model_name):
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_log_path(self, service_name):
        """Get log file path for a specific service."""
        return self.directories['logs'] / f'{service_name}.log'

# Create global config instance
config = ProjectConfig() 