import os
import requests
from PIL import Image
from io import BytesIO
import time

def download_image(url, save_path):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.resize((224, 224))
            img.save(save_path, 'JPEG')
            print(f"Downloaded: {save_path}")
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
    return False

def main():
    # More curated image URLs from various free image sources
    cat_urls = [
        # Pexels images
        "https://images.pexels.com/photos/104827/cat-pet-animal-domestic-104827.jpeg",
        "https://images.pexels.com/photos/617278/pexels-photo-617278.jpeg",
        "https://images.pexels.com/photos/1056251/pexels-photo-1056251.jpeg",
        "https://images.pexels.com/photos/2071873/pexels-photo-2071873.jpeg",
        "https://images.pexels.com/photos/1543793/pexels-photo-1543793.jpeg",
        # Pixabay images
        "https://cdn.pixabay.com/photo/2014/11/30/14/11/cat-551554_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/11/16/14/43/cat-1045782_1280.jpg",
        "https://cdn.pixabay.com/photo/2017/02/20/18/03/cat-2083492_1280.jpg",
        "https://cdn.pixabay.com/photo/2014/04/13/20/49/cat-323262_1280.jpg",
        "https://cdn.pixabay.com/photo/2015/03/27/13/16/cat-694730_1280.jpg"
    ]
    
    dog_urls = [
        # Pexels images
        "https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg",
        "https://images.pexels.com/photos/2253275/pexels-photo-2253275.jpeg",
        "https://images.pexels.com/photos/1458925/pexels-photo-1458925.jpeg",
        "https://images.pexels.com/photos/58997/pexels-photo-58997.jpeg",
        "https://images.pexels.com/photos/1490908/pexels-photo-1490908.jpeg",
        # Pixabay images
        "https://cdn.pixabay.com/photo/2016/12/13/05/15/puppy-1903313_1280.jpg",
        "https://cdn.pixabay.com/photo/2019/08/19/07/45/dog-4415649_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/02/19/15/46/dog-1210559_1280.jpg",
        "https://cdn.pixabay.com/photo/2016/05/09/10/42/weimaraner-1381186_1280.jpg",
        "https://cdn.pixabay.com/photo/2018/03/31/06/31/dog-3277416_1280.jpg"
    ]
    
    # Test images for evaluation
    test_images = {
        'cat': [
            "https://cdn.pixabay.com/photo/2017/07/25/01/22/cat-2536662_1280.jpg",
            "https://cdn.pixabay.com/photo/2016/03/28/12/35/cat-1285634_1280.jpg"
        ],
        'dog': [
            "https://cdn.pixabay.com/photo/2015/11/17/13/13/dog-1047518_1280.jpg",
            "https://cdn.pixabay.com/photo/2016/05/09/10/42/weimaraner-1381186_1280.jpg"
        ]
    }
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Download training images
    for category, urls in [('cat', cat_urls), ('dog', dog_urls)]:
        print(f"\nDownloading {category} training images...")
        category_dir = os.path.join(base_dir, 'data', category)
        os.makedirs(category_dir, exist_ok=True)
        
        for idx, url in enumerate(urls):
            save_path = os.path.join(category_dir, f"{category}_train_{idx+1}.jpg")
            download_image(url, save_path)
            time.sleep(0.5)
    
    # Download test images
    test_dir = os.path.join(base_dir, 'test_images')
    os.makedirs(test_dir, exist_ok=True)
    
    for category, urls in test_images.items():
        print(f"\nDownloading {category} test images...")
        for idx, url in enumerate(urls):
            save_path = os.path.join(test_dir, f"{category}_test_{idx+1}.jpg")
            download_image(url, save_path)
            time.sleep(0.5)

if __name__ == '__main__':
    main()
