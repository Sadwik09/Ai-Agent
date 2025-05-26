import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class SentimentModel:
    def __init__(self, vocab_size, max_len, embedding_dim=100):
        """
        Initialize the sentiment analysis model
        
        Args:
            vocab_size (int): Size of the vocabulary
            max_len (int): Maximum sequence length
            embedding_dim (int): Dimension of word embeddings
        """
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the model architecture"""
        model = Sequential([
            # Embedding layer
            Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len),
            
            # Bidirectional LSTM layers
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            
            # Dense layers
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
        """
        Train the model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs to train
        """
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                'models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test data
            y_test: Test labels
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input data
        """
        return self.model.predict(X)
    
    def save(self, filepath):
        """Save the model"""
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load a saved model"""
        model = tf.keras.models.load_model(filepath)
        return model
