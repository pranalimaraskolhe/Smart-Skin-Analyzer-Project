import tensorflow as tf
import numpy as np
from preprocess_images import preprocess_images

def evaluate_model(model_path, dataset_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Preprocess images to get validation data
    _, X_val, _, y_val = preprocess_images(dataset_path)
    
    # Evaluate the model on the validation dataset
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Example usage
model_path = 'skin_type_model.h5'
dataset_path = '/home/diot/Documents/Real-TIme-Skin-Type-Detection/skin-dataset'
evaluate_model(model_path, dataset_path)
