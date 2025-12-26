import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.xception import preprocess_input
import google.generativeai as genai

# --- Configuration ---
MODEL_PATH = 'skin_lesion_model_best.h5'
IMG_SIZE = (150, 150)
CLASS_NAMES = ['Actinic Keratosis', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic Nevi', 'Vascular Lesions']

# --- Gemini API Configuration ---
# TODO: Replace "YOUR_GEMINI_API_KEY" with the key you got from Google AI Studio
GEMINI_API_KEY = st.secrets["API_KEY"] 
genai.configure(api_key=GEMINI_API_KEY)


# --- Model Loading (with caching) ---
@st.cache_resource
def load_keras_model():
    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Keras model loaded.")
    return model

# --- Helper Functions ---
def predict(image_data, model):
    """Takes an image, preprocesses it, and returns the prediction."""
    img = image_data.resize(IMG_SIZE)
    img_array = np.asarray(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction_logits = model.predict(img_preprocessed)
    
    predicted_class_index = np.argmax(prediction_logits[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = tf.nn.softmax(prediction_logits[0])[predicted_class_index].numpy() * 100
    
    return predicted_class_name, confidence

@st.cache_data
def get_gemini_response(disease_name):
    """Sends the disease name to Gemini and gets a detailed response."""
    print(f"Requesting Gemini for: {disease_name}")
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Provide a detailed yet easy-to-understand overview of the skin condition: "{disease_name}".
    Include the following sections, using markdown for formatting:
    1.  **Overview:** A general description of what this condition is.
    2.  **Common Symptoms:** A bulleted list of typical signs.
    3.  **General Causes or Risk Factors:** What might cause it or who is at risk.
    4.  **General Treatment Approaches:** Briefly mention common ways it's treated.

    IMPORTANT: Start the entire response with this mandatory disclaimer in bold:
    '**Disclaimer: This information is for educational purposes only and is not a substitute for professional medical advice. Please consult a qualified healthcare provider for any health concerns or before making any decisions related to your health or treatment.**'
    
    Do not provide a diagnosis for a specific case or image. Keep the information general about the condition.
    """
    response = model.generate_content(prompt)
    print("Gemini response received.")
    return response.text

# --- Streamlit App Layout ---
st.set_page_config(layout="wide")
st.title('Skin Lesion Classifier & Analyzer 🩺')

col1, col2 = st.columns(2)

with col1:
    st.header("1. Upload Image")
    st.write("Upload an image of a skin lesion for classification.")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

with col2:
    st.header("2. Analysis Result")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        with st.spinner('Classifying the lesion...'):
            keras_model = load_keras_model()
            class_name, conf = predict(image, keras_model)
        
        st.success(f"Predicted Class: **{class_name}**")
        st.write(f"Confidence: **{conf:.2f}%**")

        # --- The New "Analyse" Button ---
        if st.button('Analyse Disease Details with Gemini'):
            with st.spinner('Generating detailed analysis...'):
                gemini_analysis = get_gemini_response(class_name)
                st.markdown(gemini_analysis)
    else:

        st.info("Please upload an image to see the analysis.")
