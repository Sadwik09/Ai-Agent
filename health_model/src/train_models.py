"""
Script to train and compare multiple model architectures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score

from ..models.model import AdvancedHealthModel
from ..models.advanced_models import (
    LSTMHealthModel,
    CNNHealthModel,
    EnsembleHealthModel,
    TransformerHealthModel
)
from .data_processor import AdvancedDataProcessor, create_dataloaders
from .trainer import AdvancedTrainer
from ..utils.visualization import ModelAnalyzer
from ..config.config import MODEL_CONFIG, TRAIN_CONFIG, PATH_CONFIG

def get_model(model_name, input_size):
    """Initialize model based on name."""
    models = {
        'attention': AdvancedHealthModel,
        'lstm': LSTMHealthModel,
        'cnn': CNNHealthModel,
        'ensemble': EnsembleHealthModel,
        'transformer': TransformerHealthModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    
    return models[model_name](input_size)

def optimize_hyperparameters(model, X_train, y_train, param_grid):
    """Optimize hyperparameters using GridSearchCV."""
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'roc_auc': make_scorer(roc_auc_score)
    }
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, refit='roc_auc', cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_params_, grid_search.best_score_

def train_model(model_name, train_loader, val_loader, device, save_dir):
    """Train a single model and save results."""
    print(f"\nTraining {model_name} model...")
    
    # Initialize model
    model = get_model(model_name, train_loader.dataset.features.shape[1])
    model = model.to(device)
    
    # Hyperparameter optimization
    param_grid = {
        'learning_rate': [0.001, 0.01, 0.1],
        'weight_decay': [1e-5, 1e-4, 1e-3]
    }
    best_params, best_score = optimize_hyperparameters(model, train_loader.dataset.features, train_loader.dataset.labels, param_grid)
    print(f"Best parameters: {best_params}, Best score: {best_score}")
    
    # Initialize training components
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    
    # Initialize trainer
    trainer = AdvancedTrainer(model, criterion, optimizer, device)
    
    # Train model
    trainer.train(train_loader, val_loader)
    
    # Save model and results
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model checkpoint
    trainer.save_checkpoint(
        trainer.history['val_loss'][-1],
        os.path.join(model_dir, 'model.pth')
    )
    
    # Save training history
    with open(os.path.join(model_dir, 'history.json'), 'w') as f:
        json.dump(trainer.history, f)
    
    return model, trainer.history

def evaluate_model(model, val_loader, device):
    """Evaluate model performance."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return np.array(all_preds), np.array(all_labels)

def compare_models(models_results, val_loader, device):
    """Compare performance of different models."""
    results = {}
    
    for model_name, (model, history) in models_results.items():
        print(f"\nEvaluating {model_name} model...")
        
        # Get predictions
        preds, labels = evaluate_model(model, val_loader, device)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
        
        results[model_name] = {
            'accuracy': accuracy_score(labels, preds > 0.5),
            'auc': roc_auc_score(labels, preds),
            'precision': precision_score(labels, preds > 0.5),
            'recall': recall_score(labels, preds > 0.5),
            'f1': f1_score(labels, preds > 0.5),
            'val_loss': history['val_loss'][-1],
            'val_auc': history['val_auc'][-1]
        }
    
    return results

def main():
    # Setup directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('results', timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    # List of models to train
    model_names = ['attention', 'lstm', 'cnn', 'transformer', 'ensemble']
    
    # Train all models
    models_results = {}
    for model_name in model_names:
        model, history = train_model(
            model_name,
            train_loader,
            val_loader,
            device,
            save_dir
        )
        models_results[model_name] = (model, history)
    
    # Compare models
    print("\nComparing model performance...")
    results = compare_models(models_results, val_loader, device)
    
    # Save comparison results
    with open(os.path.join(save_dir, 'model_comparison.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print comparison results
    print("\nModel Comparison Results:")
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Model:")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
    
    # Create visualizations for each model
    for model_name, (model, history) in models_results.items():
        print(f"\nGenerating visualizations for {model_name} model...")
        analyzer = ModelAnalyzer(model, device)
        
        # Plot training history
        analyzer.plot_training_history(history)
        
        # Get predictions
        preds, labels = evaluate_model(model, val_loader, device)
        
        # Plot evaluation metrics
        analyzer.plot_confusion_matrix(labels, preds > 0.5)
        analyzer.plot_roc_curve(labels, preds)
        analyzer.plot_prediction_distribution(labels, preds)
        analyzer.plot_calibration_curve(labels, preds)
        
        # Analyze feature importance
        analyzer.analyze_feature_importance(
            data_processor.feature_names,
            val_loader
        )
    
    print("\nTraining and analysis complete!")
    print(f"Results saved in: {save_dir}")

if __name__ == '__main__':
    main() 