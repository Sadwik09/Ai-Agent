import os
import sys
import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf

def load_model():
    """Load the pre-trained model."""
    try:
        model = tf.keras.models.load_model('models/catdog_model.h5')
        return model
    except:
        print("Model not found. Using a placeholder model.")
        return None

def predict_image(image):
    """Make prediction on the input image."""
    if image is None:
        return "Please upload an image"
    
    try:
        # Preprocess the image
        img = Image.fromarray(image)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        model = load_model()
        if model is None:
            return "Model not available"
        
        prediction = model.predict(img_array)
        label = "Cat" if prediction[0][0] < 0.5 else "Dog"
        confidence = 1 - prediction[0][0] if label == "Cat" else prediction[0][0]
        
        return f"{label} (Confidence: {confidence:.2%})"
    except Exception as e:
        return f"Error: {str(e)}"

def create_interface():
    """Create the Gradio interface."""
    interface = gr.Interface(
        fn=predict_image,
        inputs=gr.Image(),
        outputs=gr.Label(),
        title="Cat and Dog Classifier",
        description="Upload an image of a cat or dog to classify it.",
        examples=[
            ["test_images/cat1.jpg"],
            ["test_images/dog1.jpg"]
        ]
    )
    return interface

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("test_images", exist_ok=True)
    
    # Create and launch the interface
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860) 