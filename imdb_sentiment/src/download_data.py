import os
import kaggle
import zipfile
import shutil

def download_imdb_dataset():
    """Download the IMDB dataset from Kaggle"""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download dataset
    print("Downloading IMDB dataset...")
    kaggle.api.dataset_download_files(
        'lakshmi25npathi/imdb-dataset-of-50k-movie-reviews',
        path='data',
        unzip=True
    )
    
    # Move the file to the correct location
    if os.path.exists('data/IMDB Dataset.csv'):
        print("Dataset downloaded successfully!")
    else:
        print("Error: Dataset not found. Please download manually from:")
        print("https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")

if __name__ == '__main__':
    download_imdb_dataset() 