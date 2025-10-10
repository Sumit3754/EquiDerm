import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input

# --- Configuration ---
MODEL_PATH = 'skin_lesion_model_best.h5'

# Corrected path with no leading slash
IMAGE_PATH = 'images_to_predict/ISIC_0029306.jpg' 

# The size your model expects
IMG_SIZE = (150, 150)

# The class names MUST match the training folder names
CLASS_NAMES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

# --- Prediction ---
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print(f"Loading and preparing image from: {IMAGE_PATH}")
# Load the image
img = image.load_img(IMAGE_PATH, target_size=IMG_SIZE)

# Convert image to a numpy array and expand dimensions
img_array = np.expand_dims(image.img_to_array(img), axis=0)

# Preprocess the image for the Xception model
img_preprocessed = preprocess_input(img_array)

# Make a prediction
prediction = model.predict(img_preprocessed)

# Decode the prediction
predicted_class_index = np.argmax(prediction[0])
predicted_class_name = CLASS_NAMES[predicted_class_index]
confidence = tf.nn.softmax(prediction[0])[predicted_class_index] * 100

print("\n--- Prediction Result ---")
print(f"Predicted Class: {predicted_class_name.upper()}")
print(f"Confidence: {confidence:.2f}%")
print("-----------------------")