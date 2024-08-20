import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_images(image_folder, img_size=(224, 224)):
    images = []
    labels = []
    skin_types = {'oily': 0, 'dry': 1}  # Excluding 'hydrated'

    # Iterate over each skin type folder
    for label, index in skin_types.items():
        folder_path = os.path.join(image_folder, label)
        print(f"Checking folder: {folder_path}")
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder {folder_path} does not exist.")
            continue
        # Iterate over each image in the skin type folder
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            print(f"Reading image: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Image {img_path} could not be read.")
                continue
            img = cv2.resize(img, img_size)  # Resize image to match model input
            img = img.astype('float32') / 255.0  # Normalize pixel values
            images.append(img)
            labels.append(index)

    X = np.array(images)
    y = np.array(labels)

    if len(X) == 0:
        raise ValueError("No images found in the dataset. Check folder paths and image files.")

    # Split the dataset into training and validation sets
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Example usage
dataset_path = '/home/diot/Documents/Real-TIme-Skin-Type-Detection/skin-dataset'  # Path to your dataset folder
X_train, X_val, y_train, y_val = preprocess_images(dataset_path)

print(f"Number of training samples: {len(X_train)}")
print(f"Number of validation samples: {len(X_val)}")
