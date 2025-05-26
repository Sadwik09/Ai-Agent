import os
import numpy as np
from PIL import Image

def create_sample_image(size=(224, 224), is_cat=True):
    # Create a simple synthetic image for testing
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    if is_cat:
        # Create a cat-like shape (triangular ears)
        color = (200, 150, 100)  # Brownish color
        # Head
        img[50:170, 62:162] = color
        # Ears
        for i in range(30):
            img[20+i:21+i, 62+i:82+i] = color  # Left ear
            img[20+i:21+i, 142-i:162-i] = color  # Right ear
    else:
        # Create a dog-like shape (floppy ears)
        color = (150, 100, 50)  # Different brown
        # Head
        img[50:170, 62:162] = color
        # Ears
        img[50:90, 42:72] = color  # Left ear
        img[50:90, 152:182] = color  # Right ear
    
    return Image.fromarray(img)

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Generate 10 samples for each category
    for category, is_cat in [('cat', True), ('dog', False)]:
        category_dir = os.path.join(base_dir, 'data', category)
        os.makedirs(category_dir, exist_ok=True)
        
        for i in range(10):
            img = create_sample_image(is_cat=is_cat)
            save_path = os.path.join(category_dir, f'{category}_{i+1}.jpg')
            img.save(save_path)
            print(f"Created: {save_path}")

if __name__ == '__main__':
    main()
