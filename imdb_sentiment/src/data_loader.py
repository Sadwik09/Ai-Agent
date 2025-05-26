import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

class IMDBDataLoader:
    def __init__(self, max_words=10000, max_len=200):
        """
        Initialize the data loader
        
        Args:
            max_words (int): Maximum number of words in vocabulary
            max_len (int): Maximum sequence length
        """
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=max_words)
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and preprocess text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)
    
    def load_data(self, data_path):
        """
        Load and preprocess the IMDB dataset
        
        Args:
            data_path (str): Path to the dataset CSV file
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Read data
        df = pd.read_csv(data_path)
        
        # Clean reviews
        print("Cleaning reviews...")
        df['cleaned_review'] = df['review'].apply(self.clean_text)
        
        # Convert sentiment to binary
        df['sentiment'] = (df['sentiment'] == 'positive').astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'],
            df['sentiment'],
            test_size=0.2,
            random_state=42
        )
        
        # Fit tokenizer on training data
        print("Fitting tokenizer...")
        self.tokenizer.fit_on_texts(X_train)
        
        # Convert texts to sequences
        print("Converting texts to sequences...")
        X_train_seq = self.tokenizer.texts_to_sequences(X_train)
        X_test_seq = self.tokenizer.texts_to_sequences(X_test)
        
        # Pad sequences
        print("Padding sequences...")
        X_train_pad = pad_sequences(X_train_seq, maxlen=self.max_len)
        X_test_pad = pad_sequences(X_test_seq, maxlen=self.max_len)
        
        return X_train_pad, X_test_pad, y_train.values, y_test.values

    def get_word_index(self):
        """Get the word index dictionary"""
        return self.tokenizer.word_index
