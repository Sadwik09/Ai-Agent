"""
Data processing module for the health model.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from ..config.config import DATA_CONFIG
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class AdvancedDataProcessor:
    """Advanced data processing class with feature engineering and selection capabilities."""
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.pca = None
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        self.preprocessor = None
    
    def detect_categorical_features(self, data, threshold=10):
        """Detect categorical features based on unique value count."""
        categorical_cols = []
        for col in data.columns:
            if data[col].nunique() < threshold:
                categorical_cols.append(col)
        return categorical_cols

    def handle_missing_values(self, data, strategy='mean'):
        """Handle missing values using imputation."""
        imputer = SimpleImputer(strategy=strategy)
        return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    def remove_outliers(self, data, method='isolation_forest'):
        """Remove outliers using Isolation Forest."""
        if method == 'isolation_forest':
            clf = IsolationForest(contamination=0.1, random_state=42)
            data['outlier'] = clf.fit_predict(data)
            data = data[data['outlier'] == 1].drop('outlier', axis=1)
        return data

    def scale_features(self, data, method='standard'):
        """Scale features using specified method."""
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        return pd.DataFrame(self.scaler.fit_transform(data), columns=data.columns)
    
    def create_interaction_features(self, data):
        """Create interaction terms between important features."""
        for i in range(len(data.columns)):
            for j in range(i+1, len(data.columns)):
                col1, col2 = data.columns[i], data.columns[j]
                data[f'{col1}_{col2}_interaction'] = data[col1] * data[col2]
        return data
    
    def apply_pca(self, data, n_components=DATA_CONFIG['pca_components']):
        """Apply PCA for dimensionality reduction."""
        self.pca = PCA(n_components=n_components)
        return self.pca.fit_transform(data)
    
    def select_features(self, X, y, method=DATA_CONFIG['feature_selection_method']):
        """Select important features using specified method."""
        if method == 'mutual_info':
            self.feature_selector = SelectKBest(mutual_info_classif, k='all')
        elif method == 'chi2':
            self.feature_selector = SelectKBest(chi2, k='all')
        return self.feature_selector.fit_transform(X, y)
    
    def process_data(self, data_path):
        """Main data processing pipeline."""
        # Load data
        data = pd.read_csv(data_path)
        self.feature_names = data.columns.drop('target')
        
        # Detect categorical features
        self.categorical_features = self.detect_categorical_features(data)
        self.numerical_features = [col for col in data.columns if col not in self.categorical_features]
        
        # Handle missing values
        data = self.handle_missing_values(data)
        
        # Remove outliers
        data = self.remove_outliers(data)
        
        # Scale features
        data = self.scale_features(data)
        
        # Split features and labels
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Feature selection
        X_selected = self.select_features(X, y)
        
        # Apply PCA
        X_pca = self.apply_pca(X_selected)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_pca, y, 
            test_size=DATA_CONFIG['test_size'],
            random_state=DATA_CONFIG['random_state']
        )
        
        return X_train, X_val, y_train, y_val

class AdvancedHealthDataset(Dataset):
    """Custom dataset class with data augmentation capabilities."""
    
    def __init__(self, features, labels, transform=None, augment=False):
        self.features = features
        self.labels = labels
        self.transform = transform
        self.augment = augment
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        
        if self.augment:
            # Apply random noise
            if np.random.random() > DATA_CONFIG['augmentation_probability']:
                feature = feature + np.random.normal(0, 0.01, feature.shape)
            
            # Apply random masking
            if np.random.random() > DATA_CONFIG['augmentation_probability']:
                mask = np.random.random(feature.shape) > 0.1
                feature = feature * mask
        
        if self.transform:
            feature = self.transform(feature)
        
        return torch.FloatTensor(feature), torch.FloatTensor([label])

def create_dataloaders(X_train, X_val, y_train, y_val, batch_size=DATA_CONFIG['batch_size']):
    """Create training and validation dataloaders."""
    train_dataset = AdvancedHealthDataset(X_train, y_train, augment=True)
    val_dataset = AdvancedHealthDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader 