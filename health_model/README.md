# Advanced Health Prediction Model

This project implements an advanced health prediction model using PyTorch with attention mechanism and comprehensive analysis tools.

## Features

- Advanced data processing with feature engineering
- Attention-based neural network architecture
- Comprehensive training pipeline with early stopping
- Advanced visualization and analysis tools
- Feature importance analysis
- Model calibration analysis

## Project Structure

```
health_model/
├── config/
│   └── config.py           # Configuration parameters
├── data/                   # Data directory
├── models/
│   └── model.py           # Model architecture
├── src/
│   ├── data_processor.py  # Data processing utilities
│   ├── trainer.py         # Training utilities
│   └── main.py           # Main training script
├── utils/
│   └── visualization.py   # Visualization utilities
├── requirements.txt       # Project dependencies
└── README.md             # Project documentation
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your health data in the `data` directory as `health_data.csv`
2. Run the training script:
```bash
python -m src.main
```

## Model Architecture

The model uses an attention-based neural network with:
- Feature extraction layers
- Attention mechanism for feature importance
- Multiple processing layers with batch normalization
- Dropout for regularization
- Sigmoid output for binary classification

## Training Features

- Early stopping to prevent overfitting
- Learning rate scheduling
- Gradient clipping
- Comprehensive metrics tracking
- Model checkpointing

## Analysis Tools

The project includes various analysis tools:
- Training history visualization
- Confusion matrix
- ROC curve analysis
- Feature importance analysis
- Prediction distribution analysis
- Model calibration analysis

## Configuration

You can modify the model behavior by editing `config/config.py`:
- Model architecture parameters
- Training hyperparameters
- Data processing settings
- Path configurations

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- Other dependencies listed in requirements.txt 