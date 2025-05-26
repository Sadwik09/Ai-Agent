import os
import requests
from PIL import Image
from io import BytesIO

def download_image(url, save_path):
    try:
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
        print(f"Downloaded: {save_path}")
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")

# Sample image URLs (from Wikimedia Commons, public domain/CC licensed images)
sample_images = {
    'cat': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/481px-Cat03.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/320px-Felis_catus-cat_on_snow.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Cat_March_2010-1.jpg/320px-Cat_March_2010-1.jpg',
    ],
    'dog': [
        'https://upload.wikimedia.org/wikipedia/commons/thumb/2/2d/Dog_-_Canis_lupus_familiaris.jpg/320px-Dog_-_Canis_lupus_familiaris.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Cavalier_King_Charles_Spaniel_puppy.jpg/320px-Cavalier_King_Charles_Spaniel_puppy.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/Lucy_the_Dog_at_Oaken_Wood%2C_Challock.jpg/320px-Lucy_the_Dog_at_Oaken_Wood%2C_Challock.jpg',
    ]
}

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    for category, urls in sample_images.items():
        category_dir = os.path.join(base_dir, 'data', category)
        os.makedirs(category_dir, exist_ok=True)
        
        for idx, url in enumerate(urls):
            save_path = os.path.join(category_dir, f'{category}_{idx+1}.jpg')
            download_image(url, save_path)

if __name__ == '__main__':
    main()
