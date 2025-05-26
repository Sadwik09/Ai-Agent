import os
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import DataLoader
import matplotlib.pyplot as plt

def load_latest_models():
    """Load the latest version of each model type"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, 'results')
    
    # Get the latest run directory
    latest_run = max(os.listdir(results_dir))
    model_path = os.path.join(results_dir, latest_run, 'best_model.h5')
    
    return load_model(model_path)

def predict_image(model, image_path, data_loader):
    """Make prediction for a single image"""
    processed_image = data_loader.load_single_image(image_path)
    predictions = model.predict(processed_image)
    predicted_class = ['Cat', 'Dog'][np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return predicted_class, confidence

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(base_dir, 'test_images')
    
    # Load the model and create data loader
    print("Loading model...")
    model = load_latest_models()
    data_loader = DataLoader(data_dir=None)
    
    # Test all images in the test directory
    print("\nTesting model on test images:")
    print("-" * 50)
    
    test_images = []
    for file in os.listdir(test_dir):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_dir, file)
            true_class = 'cat' if 'cat' in file.lower() else 'dog'
            
            predicted_class, confidence = predict_image(model, image_path, data_loader)
            
            print(f"\nImage: {file}")
            print(f"True Class: {true_class.title()}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            print("-" * 50)
            
            # Save results for plotting
            test_images.append({
                'file': file,
                'true_class': true_class,
                'predicted_class': predicted_class.lower(),
                'confidence': confidence
            })
    
    # Calculate accuracy
    correct = sum(1 for img in test_images if img['true_class'] == img['predicted_class'])
    accuracy = correct / len(test_images)
    print(f"\nOverall Accuracy: {accuracy:.2%}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    for img in test_images:
        color = 'green' if img['true_class'] == img['predicted_class'] else 'red'
        plt.bar(img['file'], img['confidence'], color=color)
    
    plt.title('Model Predictions on Test Images')
    plt.xlabel('Image')
    plt.ylabel('Confidence')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(base_dir, 'test_results.png')
    plt.savefig(plot_path)
    print(f"\nTest results plot saved to: {plot_path}")

if __name__ == '__main__':
    main()
