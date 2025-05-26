import cv2
import numpy as np
from ensemble_model import EnsembleModel, find_model_paths
from data_loader import DataLoader
import tensorflow as tf
import time
import csv
import os
from datetime import datetime

class RealtimeDetector:
    def __init__(self):
        # Initialize models
        print("Loading models...")
        model_paths = find_model_paths()
        self.ensemble = EnsembleModel(model_paths)
        self.data_loader = DataLoader(data_dir=None)
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise ValueError("Could not open webcam")
            
        # Initialize CSV logging
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.csv_path = os.path.join(self.base_dir, 'realtime_predictions.csv')
        self.init_csv()
        
        # Performance tracking
        self.fps_history = []
        self.prediction_history = []
        
    def init_csv(self):
        """Initialize CSV file for logging predictions"""
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Predicted_Class', 'Confidence', 'FPS'])
    
    def log_prediction(self, prediction, confidence, fps):
        """Log prediction to CSV"""
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now(), prediction, confidence, fps])
    
    def process_frame(self, frame):
        """Process a single frame"""
        # Resize frame to model input size
        resized = cv2.resize(frame, (224, 224))
        # Convert to RGB (from BGR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # Preprocess
        processed = self.data_loader.preprocess_image(rgb)
        return processed
    
    def draw_overlay(self, frame, prediction, confidence, fps):
        """Draw prediction overlay on frame"""
        # Draw prediction
        text = f"{prediction}: {confidence:.1%}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2)
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        
        # Draw confidence bar
        bar_width = int(confidence * 200)
        cv2.rectangle(frame, (10, 70), (210, 90), (0, 0, 0), 2)
        cv2.rectangle(frame, (10, 70), (10 + bar_width, 90),
                     (0, 255, 0), -1)
        
        return frame
    
    def run(self):
        """Run real-time detection"""
        print("Starting real-time detection... Press 'q' to quit")
        
        try:
            while True:
                start_time = time.time()
                
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Process frame
                processed = self.process_frame(frame)
                
                # Make prediction
                pred = self.ensemble.predict(processed)
                pred_class = ['Cat', 'Dog'][np.argmax(pred[0])]
                confidence = float(np.max(pred[0]))
                
                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                self.fps_history.append(fps)
                self.prediction_history.append((pred_class, confidence))
                
                # Draw overlay
                frame = self.draw_overlay(frame, pred_class, confidence, fps)
                
                # Log prediction
                self.log_prediction(pred_class, confidence, fps)
                
                # Show frame
                cv2.imshow('Cat vs Dog Detector', frame)
                
                # Break on 'q' press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.save_session_stats()
    
    def save_session_stats(self):
        """Save statistics from the detection session"""
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            print(f"\nSession Statistics:")
            print(f"Average FPS: {avg_fps:.1f}")
            
            # Count predictions
            predictions = [p[0] for p in self.prediction_history]
            for cls in ['Cat', 'Dog']:
                count = predictions.count(cls)
                avg_conf = np.mean([p[1] for p in self.prediction_history 
                                  if p[0] == cls]) if count > 0 else 0
                print(f"{cls} predictions: {count} (avg confidence: {avg_conf:.1%})")

def main():
    detector = RealtimeDetector()
    detector.run()

if __name__ == '__main__':
    main()
