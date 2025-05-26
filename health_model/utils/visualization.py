"""
Visualization and analysis utilities for the health model.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import torch
import shap

class ModelAnalyzer:
    """Class for model analysis and visualization."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def plot_training_history(self, history):
        """Plot training history metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        axes[0, 0].plot(history['train_loss'], label='Train Loss')
        axes[0, 0].plot(history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Plot accuracies
        axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # Plot AUC
        axes[1, 0].plot(history['val_auc'], label='Val AUC')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        
        # Plot learning rate
        axes[1, 1].plot(history['learning_rates'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def plot_roc_curve(self, y_true, y_pred_proba):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()
    
    def analyze_feature_importance(self, feature_names, dataloader):
        """Analyze and plot feature importance using attention weights."""
        self.model.eval()
        attention_weights = []
        
        with torch.no_grad():
            for features, _ in dataloader:
                features = features.to(self.device)
                weights = self.model.get_attention_weights(features)
                attention_weights.append(weights.cpu().numpy())
        
        # Average attention weights across batches
        avg_weights = np.mean(np.concatenate(attention_weights, axis=0), axis=0)
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(avg_weights)), avg_weights)
        plt.xticks(range(len(avg_weights)), feature_names, rotation=45, ha='right')
        plt.title('Feature Importance Analysis')
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.tight_layout()
        plt.show()
        
        return avg_weights
    
    def plot_prediction_distribution(self, y_true, y_pred_proba):
        """Plot distribution of predictions for each class."""
        plt.figure(figsize=(10, 6))
        sns.kdeplot(y_pred_proba[y_true == 0], label='Negative Class')
        sns.kdeplot(y_pred_proba[y_true == 1], label='Positive Class')
        plt.title('Prediction Distribution by Class')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
    
    def plot_calibration_curve(self, y_true, y_pred_proba, n_bins=10):
        """Plot calibration curve."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate mean predicted probability and true fraction for each bin
        mean_pred = []
        true_fraction = []
        
        for i in range(n_bins):
            mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
            if np.sum(mask) > 0:
                mean_pred.append(np.mean(y_pred_proba[mask]))
                true_fraction.append(np.mean(y_true[mask]))
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_pred, true_fraction, 's-', label='Model')
        plt.plot([0, 1], [0, 1], '--', label='Perfect Calibration')
        plt.title('Calibration Curve')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('True Fraction of Positives')
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_predictions(self, predictions):
        """Analyze model predictions."""
        return {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions)
        }

def display_user_feature_importance(feature_weights, feature_names):
    """Display and plot user-provided feature importances."""
    print("\nUser-Provided Feature Importances:")
    for weight, name in zip(feature_weights, feature_names):
        print(f"{name}: {weight[0]:.4f} ± {weight[1]:.4f}")
    # Plot
    weights = np.array([w[0] for w in feature_weights])
    errors = np.array([w[1] for w in feature_weights])
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, weights, xerr=errors, color='skyblue')
    plt.xlabel('Importance Weight')
    plt.title('User-Provided Feature Importances')
    plt.tight_layout()
    plt.show()

def explain_predictions(model, input_data, feature_names):
    """Simple feature importance explanation."""
    importance = np.abs(input_data)
    return dict(zip(feature_names, importance)) 