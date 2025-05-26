import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_test_health_data(n_samples=100):
    """Generate realistic test health data."""
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamps
    base_time = datetime.now()
    timestamps = [base_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Generate realistic health features
    data = {
        'timestamp': timestamps,
        'feature1': np.random.normal(75, 5, n_samples),  # Heart Rate
        'feature2': np.random.normal(120, 10, n_samples),  # Blood Pressure
        'feature3': np.random.normal(37, 0.5, n_samples),  # Temperature
        'feature4': np.random.normal(98, 1, n_samples),  # Oxygen Level
        'feature5': np.random.normal(16, 2, n_samples),  # Respiratory Rate
        'feature6': np.random.normal(100, 10, n_samples),  # Blood Glucose
        'feature7': np.random.normal(7.4, 0.1, n_samples),  # pH Level
        'feature8': np.random.normal(140, 5, n_samples),  # Sodium Level
        'feature9': np.random.normal(4, 0.5, n_samples),  # Potassium Level
    }
    
    # Add some correlations between features
    data['feature1'] = data['feature1'] + 0.3 * data['feature2']  # Heart rate affected by BP
    data['feature4'] = data['feature4'] - 0.2 * data['feature5']  # O2 level affected by resp rate
    
    # Generate labels (1 for healthy, 0 for unhealthy)
    # Using a combination of features to determine health status
    health_score = (
        (data['feature1'] - 75) / 5 +  # Heart rate contribution
        (data['feature2'] - 120) / 10 +  # Blood pressure contribution
        (data['feature3'] - 37) / 0.5 +  # Temperature contribution
        (data['feature4'] - 98) / 1 +  # Oxygen level contribution
        (data['feature5'] - 16) / 2 +  # Respiratory rate contribution
        (data['feature6'] - 100) / 10 +  # Blood glucose contribution
        (data['feature7'] - 7.4) / 0.1 +  # pH level contribution
        (data['feature8'] - 140) / 5 +  # Sodium level contribution
        (data['feature9'] - 4) / 0.5  # Potassium level contribution
    ) / 9  # Normalize by number of features
    
    # Convert health score to binary labels with some noise
    labels = (health_score > 0).astype(int)
    labels = np.where(np.random.random(n_samples) < 0.1, 1 - labels, labels)  # Add 10% noise
    
    # Add labels to the data
    data['label'] = labels
    
    return pd.DataFrame(data)

def generate_test_batch_data(n_samples=1000):
    """Generate a larger batch of test data for batch prediction testing."""
    return generate_test_health_data(n_samples)

def generate_test_realtime_data(n_samples=100):
    """Generate test data for real-time monitoring."""
    np.random.seed(42)
    
    metrics = {
        'Heart Rate': {'mean': 75, 'std': 5, 'unit': 'bpm'},
        'Blood Pressure': {'mean': 120, 'std': 10, 'unit': 'mmHg'},
        'Temperature': {'mean': 37, 'std': 0.5, 'unit': '°C'},
        'Oxygen Level': {'mean': 98, 'std': 1, 'unit': '%'},
        'Respiratory Rate': {'mean': 16, 'std': 2, 'unit': 'breaths/min'},
        'Blood Glucose': {'mean': 100, 'std': 10, 'unit': 'mg/dL'}
    }
    
    data = []
    base_time = datetime.now()
    
    for i in range(n_samples):
        current_time = base_time + timedelta(seconds=i)
        for metric, params in metrics.items():
            value = np.random.normal(params['mean'], params['std'])
            data.append({
                'timestamp': current_time,
                'value': value,
                'metric': metric,
                'unit': params['unit']
            })
    
    df = pd.DataFrame(data)
    # Ensure all required columns are present
    required_columns = ['timestamp', 'value', 'metric', 'unit']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df

if __name__ == "__main__":
    # Generate and save test data
    test_data = generate_test_health_data()
    test_data.to_csv('test_health_data.csv', index=False)
    
    batch_data = generate_test_batch_data()
    batch_data.to_csv('test_batch_data.csv', index=False)
    
    realtime_data = generate_test_realtime_data()
    realtime_data.to_csv('test_realtime_data.csv', index=False)
    
    print("Test data generated and saved successfully!") 