import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from data_loader import IMDBDataLoader
from model import SentimentModel

def plot_training_history(history, save_path):
    """Plot training history"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train IMDB sentiment analysis model')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the IMDB dataset')
    parser.add_argument('--max_words', type=int, default=10000, help='Maximum number of words in vocabulary')
    parser.add_argument('--max_len', type=int, default=200, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of word embeddings')
    args = parser.parse_args()
    
    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join('results', timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    data_loader = IMDBDataLoader(max_words=args.max_words, max_len=args.max_len)
    X_train, X_test, y_train, y_test = data_loader.load_data(args.data_path)
    
    # Split training data into train and validation sets
    val_size = int(0.1 * len(X_train))
    X_val = X_train[-val_size:]
    y_val = y_train[-val_size:]
    X_train = X_train[:-val_size]
    y_train = y_train[:-val_size]
    
    # Create and train model
    print("Creating and training model...")
    model = SentimentModel(
        vocab_size=args.max_words,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim
    )
    
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Plot training history
    plot_training_history(history, os.path.join(results_dir, 'training_history.png'))
    
    # Save model
    model.save(os.path.join('models', 'best_model.h5'))
    print("Model saved successfully!")

if __name__ == '__main__':
    main()
