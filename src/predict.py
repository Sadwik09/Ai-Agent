import os
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from data_loader import DataLoader

class Predictor:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.data_loader = DataLoader(data_dir=None)  # We only need it for preprocessing
        self.classes = ['Cat', 'Dog']
    
    def predict_image(self, image):
        # Preprocess the image
        processed_image = self.data_loader.load_single_image(image)
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        predicted_class = self.classes[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return {
            predicted_class: confidence,
            self.classes[1-self.classes.index(predicted_class)]: 1-confidence
        }

def create_gradio_interface(model_path):
    predictor = Predictor(model_path)
    
    iface = gr.Interface(
        fn=predictor.predict_image,
        inputs=gr.Image(type="filepath"),
        outputs=gr.Label(num_top_classes=2),
        title="Cat vs Dog Classifier",
        description="Upload an image of a cat or dog to classify it!",
        examples=[
            ["examples/cat1.jpg"],
            ["examples/dog1.jpg"],
        ]
    )
    
    return iface

if __name__ == "__main__":
    # Use the latest trained model
    results_dir = "results"
    latest_run = max(os.listdir(results_dir))
    model_path = os.path.join(results_dir, latest_run, "best_model.h5")
    
    # Create and launch the interface
    iface = create_gradio_interface(model_path)
    iface.launch(share=True)
