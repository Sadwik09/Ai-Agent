"""
Main training script for the health prediction model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import numpy as np

from ..models.model import AdvancedHealthModel
from .data_processor import AdvancedDataProcessor, create_dataloaders
from .trainer import AdvancedTrainer
from ..utils.visualization import ModelAnalyzer
from ..config.config import MODEL_CONFIG, TRAIN_CONFIG, PATH_CONFIG

def setup_directories():
    """Create necessary directories if they don't exist."""
    for path in PATH_CONFIG.values():
        if isinstance(path, str) and ('/' in path or '\\' in path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)

def main():
    # Setup directories
    setup_directories()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize data processor
    data_processor = AdvancedDataProcessor()
    
    # Process data
    print('Processing data...')
    X_train, X_val, y_train, y_val = data_processor.process_data(
        os.path.join(PATH_CONFIG['data_dir'], PATH_CONFIG['data_file'])
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X_train, X_val, y_train, y_val,
        batch_size=TRAIN_CONFIG['batch_size']
    )
    
    # Initialize model
    print('Initializing model...')
    MODEL_CONFIG['input_size'] = X_train.shape[1]
    model = AdvancedHealthModel(
        input_size=MODEL_CONFIG['input_size'],
        hidden_sizes=MODEL_CONFIG['hidden_sizes'],
        dropout_rate=MODEL_CONFIG['dropout_rate']
    ).to(device)
    
    # Initialize training components
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, criterion, optimizer, device)
    
    # Train model
    print('Starting training...')
    trainer.train(train_loader, val_loader)
    
    # Initialize analyzer
    analyzer = ModelAnalyzer(model, device)
    
    # Plot training history
    print('Generating training visualizations...')
    analyzer.plot_training_history(trainer.history)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        val_preds = []
        val_labels = []
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            val_preds.extend(outputs.cpu().numpy())
            val_labels.extend(labels.numpy())
    
    # Plot evaluation metrics
    print('Generating evaluation visualizations...')
    analyzer.plot_confusion_matrix(
        val_labels,
        (np.array(val_preds) > 0.5).astype(int)
    )
    analyzer.plot_roc_curve(val_labels, val_preds)
    analyzer.plot_prediction_distribution(val_labels, val_preds)
    analyzer.plot_calibration_curve(val_labels, val_preds)
    
    # Analyze feature importance
    print('Analyzing feature importance...')
    analyzer.analyze_feature_importance(
        data_processor.feature_names,
        val_loader
    )
    
    print('Training and analysis complete!')

if __name__ == '__main__':
    main() 