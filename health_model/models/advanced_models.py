"""
Advanced model architectures for health prediction.
"""

import torch
import torch.nn as nn
from ..config.config import MODEL_CONFIG

class LSTMHealthModel(nn.Module):
    """LSTM-based health prediction model."""
    
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super(LSTMHealthModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Reshape input for LSTM (sequence_length, batch_size, input_size)
        self.reshape = nn.Linear(input_size, input_size)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape input
        x = self.reshape(x).unsqueeze(1)  # Add sequence dimension
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)
        attended = lstm_out * attention_weights
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        
        # Final prediction
        return self.fc(pooled)

class CNNHealthModel(nn.Module):
    """CNN-based health prediction model."""
    
    def __init__(self, input_size, hidden_channels=64, dropout=0.3):
        super(CNNHealthModel, self).__init__()
        
        # Reshape input for CNN (batch_size, channels, sequence_length)
        self.reshape = nn.Linear(input_size, input_size)
        
        # CNN layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Conv1d(hidden_channels*2, hidden_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels*4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels*4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Reshape input
        x = self.reshape(x).unsqueeze(1)  # Add channel dimension
        
        # CNN forward pass
        x = self.conv_layers(x)
        x = self.pool(x).squeeze(-1)
        
        # Final prediction
        return self.fc(x)

class EnsembleHealthModel(nn.Module):
    """Ensemble of different model architectures."""
    
    def __init__(self, input_size, hidden_sizes=MODEL_CONFIG['hidden_sizes'], dropout_rate=MODEL_CONFIG['dropout_rate']):
        super(EnsembleHealthModel, self).__init__()
        
        # Initialize different models
        self.attention_model = AdvancedHealthModel(input_size, hidden_sizes, dropout_rate)
        self.lstm_model = LSTMHealthModel(input_size)
        self.cnn_model = CNNHealthModel(input_size)
        
        # Ensemble weights
        self.weights = nn.Parameter(torch.ones(3) / 3)
        
        # Softmax for weights
        self.softmax = nn.Softmax(dim=0)
    
    def forward(self, x):
        # Get predictions from each model
        att_pred = self.attention_model(x)
        lstm_pred = self.lstm_model(x)
        cnn_pred = self.cnn_model(x)
        
        # Stack predictions
        predictions = torch.stack([att_pred, lstm_pred, cnn_pred], dim=0)
        
        # Apply softmax to weights
        weights = self.softmax(self.weights)
        
        # Weighted average
        ensemble_pred = torch.sum(predictions * weights.view(-1, 1, 1), dim=0)
        
        return ensemble_pred
    
    def get_model_weights(self):
        """Get the current ensemble weights."""
        return self.softmax(self.weights).detach().cpu().numpy()

class TransformerHealthModel(nn.Module):
    """Transformer-based health prediction model."""
    
    def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dropout=0.3):
        super(TransformerHealthModel, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Add sequence dimension and embed
        x = self.embedding(x).unsqueeze(1)
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Transformer forward pass
        x = self.transformer_encoder(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Final prediction
        return self.fc(x) 