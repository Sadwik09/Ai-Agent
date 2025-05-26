import os
import requests
from PIL import Image
from io import BytesIO
import time
import json

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

def download_from_unsplash(query, save_dir, num_images=50):
    """Download images from Unsplash API"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Unsplash demo API - for development purposes
    api_url = f"https://api.unsplash.com/search/photos"
    headers = {
        "Authorization": "Client-ID YOUR_ACCESS_KEY"  # Replace with your Unsplash API key
    }
    
    params = {
        "query": query,
        "per_page": num_images,
        "orientation": "squarish"
    }
    
    try:
        response = requests.get(api_url, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            count = 0
            
            for idx, photo in enumerate(data['results']):
                if count >= num_images:
                    break
                    
                image_url = photo['urls']['regular']
                save_path = os.path.join(save_dir, f"{query.replace(' ', '_')}_{idx+1}.jpg")
                
                if download_image(image_url, save_path):
                    count += 1
                
                # Add a small delay
                time.sleep(0.5)
            
            return count
        else:
            print(f"API Error: {response.status_code}")
            return 0
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 0

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    
    # Alternative: Use a curated list of image URLs
    cat_urls = [
        "https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg",
        "https://images.pexels.com/photos/416160/pexels-photo-416160.jpeg",
        "https://images.pexels.com/photos/1056251/pexels-photo-1056251.jpeg",
        "https://images.pexels.com/photos/320014/pexels-photo-320014.jpeg",
        # Add more URLs here
    ]
    
    dog_urls = [
        "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg",
        "https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg",
        "https://images.pexels.com/photos/1390361/pexels-photo-1390361.jpeg",
        "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg",
        # Add more URLs here
    ]
    
    # Clear existing data directories
    for category in ['cat', 'dog']:
        category_dir = os.path.join(data_dir, category)
        if os.path.exists(category_dir):
            for file in os.listdir(category_dir):
                os.remove(os.path.join(category_dir, file))
    
    # Download images from direct URLs
    for category, urls in [('cat', cat_urls), ('dog', dog_urls)]:
        print(f"\nDownloading {category} images...")
        category_dir = os.path.join(data_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        count = 0
        for idx, url in enumerate(urls):
            save_path = os.path.join(category_dir, f"{category}_{idx+1}.jpg")
            if download_image(url, save_path):
                count += 1
            time.sleep(0.5)
        
        print(f"Successfully downloaded {count} {category} images")

if __name__ == '__main__':
    main()
