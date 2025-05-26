import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

class EnsembleModel:
    def __init__(self, model_paths):
        self.models = [load_model(path) for path in model_paths]
        
    def predict(self, image):
        """Make ensemble prediction using weighted voting"""
        predictions = []
        for model in self.models:
            pred = model.predict(image, verbose=0)
            predictions.append(pred)
        
        # Average predictions from all models
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred

def find_model_paths():
    """Find paths to all trained models"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    
    model_paths = []
    for run_dir in os.listdir(results_dir):
        model_path = os.path.join(results_dir, run_dir, 'best_model.h5')
        if os.path.exists(model_path):
            model_paths.append(model_path)
    
    return model_paths

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the plot
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig(os.path.join(base_dir, 'confusion_matrix.png'))
    plt.close()

def analyze_model_performance(ensemble, data_loader, test_images):
    """Analyze and visualize model performance"""
    y_true = []
    y_pred = []
    confidences = []
    
    for img_path in test_images:
        true_class = 1 if 'dog' in os.path.basename(img_path).lower() else 0
        processed_image = data_loader.load_single_image(img_path)
        pred = ensemble.predict(processed_image)
        pred_class = np.argmax(pred[0])
        confidence = float(np.max(pred[0]))
        
        y_true.append(true_class)
        y_pred.append(pred_class)
        confidences.append(confidence)
    
    # Generate classification report
    classes = ['Cat', 'Dog']
    report = classification_report(y_true, y_pred, target_names=classes)
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes)
    
    # Plot confidence distribution
    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(classes):
        class_confidences = [conf for j, conf in enumerate(confidences) if y_pred[j] == i]
        if class_confidences:
            plt.hist(class_confidences, alpha=0.5, label=cls, bins=10)
    
    plt.title('Prediction Confidence Distribution')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig(os.path.join(base_dir, 'confidence_distribution.png'))
    plt.close()
    
    return np.mean(y_true == y_pred), np.mean(confidences)
