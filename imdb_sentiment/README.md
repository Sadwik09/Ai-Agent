# IMDB Sentiment Analysis

This project implements a sentiment analysis model for IMDB movie reviews using deep learning. The model is based on a Bidirectional LSTM architecture and can classify movie reviews as positive or negative.

## Project Structure

```
imdb_sentiment/
├── data/               # Directory for dataset
├── models/            # Directory for saved models
├── notebooks/         # Jupyter notebooks
├── results/           # Training results and plots
├── src/               # Source code
│   ├── data_loader.py # Data loading and preprocessing
│   ├── model.py       # Model architecture
│   ├── train.py       # Training script
│   └── predict.py     # Prediction script
└── requirements.txt   # Project dependencies
```

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the IMDB dataset:
- Visit [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Download the dataset and place it in the `data` directory
- The dataset should be in CSV format with 'review' and 'sentiment' columns

## Usage

### Training

To train the model:

```bash
python src/train.py --data_path data/imdb_dataset.csv --max_words 10000 --max_len 200 --batch_size 32 --epochs 10
```

Arguments:
- `--data_path`: Path to the IMDB dataset CSV file
- `--max_words`: Maximum number of words in vocabulary (default: 10000)
- `--max_len`: Maximum sequence length (default: 200)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of epochs to train (default: 10)
- `--embedding_dim`: Dimension of word embeddings (default: 100)

### Prediction

For single text prediction:

```bash
python src/predict.py --model_path models/best_model.h5 --text "This movie was fantastic!"
```

For batch prediction:

```bash
python src/predict.py --model_path models/best_model.h5 --input_file data/test_reviews.csv --output_file results/predictions.csv
```

Arguments:
- `--model_path`: Path to the trained model
- `--text`: Single text for prediction
- `--input_file`: Path to input CSV file for batch prediction
- `--output_file`: Path to output CSV file for batch prediction

## Model Architecture

The model uses a Bidirectional LSTM architecture:
1. Embedding layer
2. Bidirectional LSTM layers
3. Dropout layers for regularization
4. Dense layers for classification

## Results

Training results including accuracy and loss plots are saved in the `results` directory. Each training run creates a timestamped directory containing:
- Training history plot
- Model evaluation metrics
- Best model weights

## License

This project is licensed under the MIT License - see the LICENSE file for details.
