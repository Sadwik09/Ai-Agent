import os
import argparse
import numpy as np
import pandas as pd
from data_loader import IMDBDataLoader
from model import SentimentModel

def predict_sentiment(text, model, data_loader):
    """
    Predict sentiment for a single text
    
    Args:
        text (str): Input text
        model: Trained model
        data_loader: DataLoader instance
    
    Returns:
        float: Sentiment score (0-1)
    """
    # Clean and preprocess text
    cleaned_text = data_loader.clean_text(text)
    
    # Convert to sequence
    sequence = data_loader.tokenizer.texts_to_sequences([cleaned_text])
    
    # Pad sequence
    padded_sequence = data_loader.pad_sequences(sequence, maxlen=data_loader.max_len)
    
    # Make prediction
    prediction = model.predict(padded_sequence)[0][0]
    
    return prediction

def batch_predict(input_file, output_file, model, data_loader):
    """
    Make predictions for a batch of texts
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output CSV file
        model: Trained model
        data_loader: DataLoader instance
    """
    # Read input data
    df = pd.read_csv(input_file)
    
    # Make predictions
    predictions = []
    for text in df['review']:
        pred = predict_sentiment(text, model, data_loader)
        predictions.append(pred)
    
    # Add predictions to dataframe
    df['sentiment_score'] = predictions
    df['predicted_sentiment'] = (df['sentiment_score'] > 0.5).astype(int)
    
    # Save results
    df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Make sentiment predictions')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_file', type=str, help='Path to input CSV file for batch prediction')
    parser.add_argument('--output_file', type=str, help='Path to output CSV file for batch prediction')
    parser.add_argument('--text', type=str, help='Single text for prediction')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    model = SentimentModel.load(args.model_path)
    
    # Initialize data loader
    data_loader = IMDBDataLoader()
    
    if args.input_file and args.output_file:
        # Batch prediction
        print("Making batch predictions...")
        batch_predict(args.input_file, args.output_file, model, data_loader)
    elif args.text:
        # Single text prediction
        print("Making prediction for single text...")
        prediction = predict_sentiment(args.text, model, data_loader)
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        print(f"\nText: {args.text}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {prediction:.2%}")
    else:
        print("Please provide either --input_file and --output_file for batch prediction or --text for single prediction")

if __name__ == '__main__':
    main()
