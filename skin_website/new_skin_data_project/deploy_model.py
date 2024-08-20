from tensorflow.keras.models import load_model
import cv2
import numpy as np

def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, img_size)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_skin_type(image_path):
    model = load_model('skin_type_model.h5')
    img = preprocess_image(image_path)
    prediction = model.predict(img)
    skin_types = ['oily', 'dry']  # Updated to match the two classes
    predicted_class = np.argmax(prediction, axis=1)[0]
    return skin_types[predicted_class]

# Example usage
skin_type = predict_skin_type('/home/diot/Documents/skin_detect/image.jpg')
print(f"Predicted Skin Type: {skin_type}")
