"""
Model architecture for the health prediction model.
"""

import torch
import torch.nn as nn
import sys
import os

# Add the parent directory to the system path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config.config import MODEL_CONFIG  # Changed to relative import

class AttentionLayer(nn.Module):
    """Attention mechanism for feature importance weighting."""
    
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights

class AdvancedHealthModel(nn.Module):
    """Advanced health prediction model with attention mechanism."""
    
    def __init__(self, input_size=9):
        super(AdvancedHealthModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x
    
    def get_attention_weights(self, x):
        """Get attention weights for feature importance analysis."""
        x = x.unsqueeze(0)
        _, weights = self.attention(x, x, x)
        return weights.squeeze(0) 