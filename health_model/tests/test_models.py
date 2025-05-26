import unittest
import torch
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add the parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import AdvancedHealthModel
from src.app import (
    predict, simple_chatbot, advanced_chatbot, gemini_chatbot,
    generate_realtime_data, create_realtime_chart,
    create_prediction_chart, create_feature_importance_chart,
    create_confidence_chart
)

class TestHealthModels(unittest.TestCase):
    def setUp(self):
        """Set up test data and models."""
        self.input_size = 9
        self.model = AdvancedHealthModel(input_size=self.input_size)
        self.test_input = torch.randn(1, self.input_size)
        self.test_features = np.random.randn(1, self.input_size)
        
    def test_model_architecture(self):
        """Test the model architecture."""
        # Test model initialization
        self.assertEqual(self.model.layer1.in_features, self.input_size)
        self.assertEqual(self.model.layer1.out_features, 64)
        self.assertEqual(self.model.layer4.out_features, 1)
        
        # Test forward pass
        output = self.model(self.test_input)
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(torch.all(output >= 0) and torch.all(output <= 1))
    
    def test_prediction_function(self):
        """Test the prediction function."""
        prediction = predict(self.model, self.test_features)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.shape, (1, 1))
        self.assertTrue(np.all(prediction >= 0) and np.all(prediction <= 1))
    
    def test_chatbots(self):
        """Test all chatbot implementations."""
        # Test simple chatbot
        test_messages = [
            "hello",
            "how to predict",
            "thank you",
            "what are my symptoms",
            "goodbye"
        ]
        for msg in test_messages:
            response = simple_chatbot(msg)
            self.assertIsInstance(response, str)
            self.assertTrue(len(response) > 0)
        
        # Test advanced chatbot (OpenAI)
        try:
            response = advanced_chatbot("Hello")
            self.assertIsInstance(response, str)
        except Exception as e:
            print(f"OpenAI API test skipped: {str(e)}")
        
        # Test Gemini chatbot
        try:
            response = gemini_chatbot("Hello")
            self.assertIsInstance(response, str)
        except Exception as e:
            print(f"Gemini API test skipped: {str(e)}")
    
    def test_realtime_data(self):
        """Test real-time data generation."""
        data = generate_realtime_data()
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(len(data) > 0)
        required_columns = ['timestamp', 'value', 'metric', 'unit']
        self.assertTrue(all(col in data.columns for col in required_columns))
        
        # Test unit values
        units = {
            'Heart Rate': 'bpm',
            'Blood Pressure': 'mmHg',
            'Temperature': '°C',
            'Oxygen Level': '%',
            'Respiratory Rate': 'breaths/min',
            'Blood Glucose': 'mg/dL'
        }
        
        for metric, expected_unit in units.items():
            metric_data = data[data['metric'] == metric]
            if not metric_data.empty:
                self.assertEqual(metric_data['unit'].iloc[0], expected_unit)
    
    def test_visualization_functions(self):
        """Test visualization functions."""
        # Create test data
        test_data = pd.DataFrame({
            'timestamp': [datetime.now() for _ in range(10)],
            'value': np.random.randn(10),
            'metric': ['Heart Rate'] * 10,
            'unit': ['bpm'] * 10
        })
        
        # Test realtime chart
        realtime_chart = create_realtime_chart(test_data)
        self.assertIsNotNone(realtime_chart)
        
        # Test prediction chart
        prediction_data = pd.DataFrame({
            'timestamp': [datetime.now() for _ in range(10)],
            'prediction': np.random.rand(10),
            'confidence': np.random.rand(10)
        })
        pred_chart = create_prediction_chart(prediction_data)
        self.assertIsNotNone(pred_chart)
        
        # Test feature importance chart
        feature_data = pd.DataFrame({
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'feature3': np.random.rand(10),
            'feature4': np.random.rand(10),
            'feature5': np.random.rand(10),
            'feature6': np.random.rand(10),
            'feature7': np.random.rand(10),
            'feature8': np.random.rand(10),
            'feature9': np.random.rand(10),
            'prediction': np.random.rand(10),
            'confidence': np.random.rand(10)
        })
        feature_chart = create_feature_importance_chart(feature_data)
        self.assertIsNotNone(feature_chart)
        
        # Test confidence chart
        conf_chart = create_confidence_chart(prediction_data)
        self.assertIsNotNone(conf_chart)

def run_tests():
    """Run all tests."""
    unittest.main()

if __name__ == '__main__':
    run_tests() 