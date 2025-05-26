"""
Script to train models on all datasets.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import fetch_openml
import requests
import zipfile
import io

# Add the parent directory to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.model import AdvancedHealthModel
from utils.visualization import ModelAnalyzer, explain_predictions

def download_dataset(dataset_name):
    """Download the specified dataset."""
    if dataset_name == 'heart_disease':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(io.StringIO(response.text), header=None)
            data.to_csv('health_model/data/heart_disease.csv', index=False)
            print(f"Saved Heart Disease dataset to health_model/data/heart_disease.csv")
        else:
            print(f"Error downloading heart_disease dataset: HTTP Error {response.status_code}")
    elif dataset_name == 'diabetes':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data'
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(io.StringIO(response.text), header=None)
            data.to_csv('health_model/data/diabetes.csv', index=False)
            print(f"Saved Diabetes dataset to health_model/data/diabetes.csv")
        else:
            print(f"Error downloading diabetes dataset: HTTP Error {response.status_code}")
    elif dataset_name == 'breast_cancer':
        try:
            data = fetch_openml(name='breast-cancer-wisconsin', version=1, as_frame=True)
            data.to_csv('health_model/data/breast_cancer.csv', index=False)
            print(f"Saved Breast Cancer dataset to health_model/data/breast_cancer.csv")
        except Exception as e:
            print(f"Error downloading breast_cancer dataset: {e}")
    elif dataset_name == 'stroke':
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00519/stroke_data.csv'
        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(io.StringIO(response.text))
            data.to_csv('health_model/data/stroke.csv', index=False)
            print(f"Saved Stroke Prediction dataset to health_model/data/stroke.csv")
        else:
            print(f"Error downloading stroke dataset: HTTP Error {response.status_code}")

def train_on_dataset(dataset_name):
    """Train models on the specified dataset."""
    print(f"\nProcessing {dataset_name} dataset...")
    data = pd.read_csv(f'health_model/data/{dataset_name}.csv')
    
    # Handle missing values
    data = data.replace('?', np.nan)
    data = data.dropna()
    
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    model = AdvancedHealthModel(input_size=X_train.shape[1])
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.FloatTensor(y_train).unsqueeze(1))
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/100], Loss: {loss.item():.4f}')
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test)).numpy()
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

def main():
    """Main function to train models on all datasets."""
    datasets = ['heart_disease', 'diabetes', 'breast_cancer', 'stroke']
    for dataset in datasets:
        download_dataset(dataset)
        train_on_dataset(dataset)

if __name__ == "__main__":
    main() 