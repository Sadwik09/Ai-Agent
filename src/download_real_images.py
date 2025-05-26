import os
import requests
from PIL import Image
from io import BytesIO
from duckduckgo_search import DDGS
import time

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize to a standard size
            img = img.resize((224, 224))
            img.save(save_path, 'JPEG')
            print(f"Downloaded: {save_path}")
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
    return False

def search_and_download(query, save_dir, num_images=50):
    """Search for images and download them"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Search for images
    with DDGS() as ddgs:
        results = list(ddgs.images(
            query,
            max_results=num_images * 2  # Get extra results in case some fail
        ))
    
    count = 0
    for idx, result in enumerate(results):
        if count >= num_images:
            break
            
        image_url = result['image']
        save_path = os.path.join(save_dir, f"{query.replace(' ', '_')}_{idx+1}.jpg")
        
        if download_image(image_url, save_path):
            count += 1
        
        # Add a small delay to be nice to the servers
        time.sleep(0.5)
    
    return count

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Clear existing data directories
    for category in ['cat', 'dog']:
        category_dir = os.path.join(data_dir, category)
        if os.path.exists(category_dir):
            for file in os.listdir(category_dir):
                os.remove(os.path.join(category_dir, file))
    
    # Download new images
    categories = {
        'cat': 'cute cat pet clear photo',
        'dog': 'cute dog pet clear photo'
    }
    
    for category, search_query in categories.items():
        print(f"\nDownloading {category} images...")
        category_dir = os.path.join(data_dir, category)
        count = search_and_download(search_query, category_dir, num_images=50)
        print(f"Successfully downloaded {count} {category} images")

if __name__ == '__main__':
    main()
