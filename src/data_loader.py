import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from tensorflow.keras.utils import to_categorical

class DataLoader:
    def __init__(self, data_dir, img_size=(224, 224), augment=True):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = ['cat', 'dog']
        self.augment = augment
        
        # Define augmentation pipeline
        self.aug_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.Blur(p=1),
                A.MotionBlur(p=1),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=1),
                A.GridDistortion(p=1),
                A.ElasticTransform(p=1),
            ], p=0.3),
            A.ColorJitter(p=0.3),
        ])

    def load_data(self, split_ratio=0.2):
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    if self.augment:
                        augmented = self.aug_pipeline(image=img)
                        img = augmented['image']
                    
                    img = img / 255.0  # Normalize
                    images.append(img)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error loading image {img_path}: {str(e)}")
                    continue

        X = np.array(images)
        y = to_categorical(np.array(labels))

        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=split_ratio, random_state=42, stratify=y
        )

        return X_train, X_val, y_train, y_val

    def load_single_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        return np.expand_dims(img, axis=0)
