"""
Script to download and prepare multiple health datasets for training.
"""

import os
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
import requests
import zipfile
import io
from pathlib import Path

class HealthDataDownloader:
    def __init__(self, data_dir='health_model/data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_heart_disease(self):
        """Download Heart Disease dataset from UCI."""
        print("Downloading Heart Disease dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
                  'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
        
        df = pd.read_csv(url, names=columns)
        df['target'] = (df['target'] > 0).astype(int)  # Convert to binary classification
        
        # Save dataset
        output_path = os.path.join(self.data_dir, 'heart_disease.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved Heart Disease dataset to {output_path}")
        return output_path
    
    def download_diabetes(self):
        """Download Pima Indians Diabetes dataset."""
        print("Downloading Diabetes dataset...")
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
        columns = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness', 
                  'insulin', 'bmi', 'diabetes_pedigree', 'age', 'target']
        
        df = pd.read_csv(url, names=columns)
        
        # Save dataset
        output_path = os.path.join(self.data_dir, 'diabetes.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved Diabetes dataset to {output_path}")
        return output_path
    
    def download_breast_cancer(self):
        """Download Breast Cancer dataset from scikit-learn."""
        print("Downloading Breast Cancer dataset...")
        data = fetch_openml(name='breast-cancer-wisconsin', version=1, as_frame=True)
        df = data.frame
        
        # Convert target to binary
        df['target'] = (df['target'] == 'malignant').astype(int)
        
        # Save dataset
        output_path = os.path.join(self.data_dir, 'breast_cancer.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved Breast Cancer dataset to {output_path}")
        return output_path
    
    def download_stroke(self):
        """Download Stroke Prediction dataset from Kaggle."""
        print("Downloading Stroke Prediction dataset...")
        url = "https://raw.githubusercontent.com/datasets/stroke-prediction/master/data/stroke.csv"
        
        df = pd.read_csv(url)
        
        # Handle categorical variables
        df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 
                                       'Residence_type', 'smoking_status'])
        
        # Save dataset
        output_path = os.path.join(self.data_dir, 'stroke.csv')
        df.to_csv(output_path, index=False)
        print(f"Saved Stroke dataset to {output_path}")
        return output_path
    
    def preprocess_dataset(self, df, target_col='target'):
        """Preprocess dataset for training."""
        # Handle missing values
        df = df.fillna(df.mean())
        
        # Separate features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled, y.values, X.columns.tolist()
    
    def download_all_datasets(self):
        """Download and prepare all datasets."""
        datasets = {
            'heart_disease': self.download_heart_disease,
            'diabetes': self.download_diabetes,
            'breast_cancer': self.download_breast_cancer,
            'stroke': self.download_stroke
        }
        
        dataset_paths = {}
        for name, download_func in datasets.items():
            try:
                path = download_func()
                dataset_paths[name] = path
            except Exception as e:
                print(f"Error downloading {name} dataset: {str(e)}")
        
        return dataset_paths

def main():
    # Initialize downloader
    downloader = HealthDataDownloader()
    
    # Download all datasets
    print("Starting dataset downloads...")
    dataset_paths = downloader.download_all_datasets()
    
    print("\nDownloaded datasets:")
    for name, path in dataset_paths.items():
        print(f"- {name}: {path}")

if __name__ == '__main__':
    main() 