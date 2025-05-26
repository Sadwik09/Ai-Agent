"""
Configuration parameters for the health model.
"""

# Model Configuration
MODEL_CONFIG = {
    'input_size': None,  # Will be set based on data
    'hidden_sizes': [256, 128, 64],
    'dropout_rate': 0.3,
    'attention_size': 64
}

# Training Configuration
TRAIN_CONFIG = {
    'batch_size': 32,
    'epochs': 100,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 5,
    'gradient_clip_value': 1.0
}

# Data Processing Configuration
DATA_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'pca_components': 0.95,  # Explained variance ratio
    'feature_selection_method': 'mutual_info',  # or 'chi2'
    'augmentation_probability': 0.5,
    'batch_size': 32
}

# Path Configuration
PATH_CONFIG = {
    'data_dir': 'data/',
    'model_dir': 'models/',
    'best_model_path': 'models/best_model.pth',
    'data_file': 'health_data.csv'
} 