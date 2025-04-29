import streamlit as st
import numpy as np
import tensorflow as tf
import pickle
import cv2
from keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import torch
import bcrypt
import json
import os

# -------------------- Load Models --------------------

@st.cache_resource
def load_eeg_model():
    model = tf.keras.models.load_model('eeg_emotion_model_final.h5')
    with open('scaler_final.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder_final.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, scaler, label_encoder

@st.cache_resource
def load_face_model():
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

@st.cache_resource
def load_face_model1():
    model = load_model('model_78.h5')
    model.load_weights("model_weights_78.h5")
    return model
@st.cache_resource
def load_speech_model():
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
    model = Wav2Vec2ForSequenceClassification.from_pretrained("r-f/wav2vec-english-speech-emotion-recognition")
    return feature_extractor, model

def load_speech_model1():
    model = load_model('speech.h5')
    model.load_weights("speech.h5")
    return model
eeg_model, scaler, label_encoder = load_eeg_model()
# face_model = load_face_model()
feature_extractor, speech_model = load_speech_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
# -------------------- Authentication --------------------

USER_FILE = 'users.json'

def load_users():
    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as file:
            return json.load(file)
    return {}

def save_users(users):
    with open(USER_FILE, 'w') as file:
        json.dump(users, file, indent=4)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

# -------------------- Signup --------------------
def predict_emotion(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_cascade = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
    
    if len(face_cascade) == 0:
        return image, "No face detected"
    
    for (x, y, w, h) in face_cascade:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        x=load_face_model1()
        prediction = x.predict(roi)[0]
        maxindex = int(np.argmax(prediction))
        finalout = emotion_labels[maxindex]
        output = str(finalout)
        
        label_position = (x, y - 10)
        cv2.putText(image, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return image, output
def signup():
    st.title("🔐 Signup")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    if st.button("Signup"):
        if username and password and confirm_password:
            if password == confirm_password:
                users = load_users()
                if username in users:
                    st.error("Username already exists!")
                else:
                    users[username] = hash_password(password)
                    save_users(users)
                    st.success("Signup successful! Please login.")
                    st.session_state['registered'] = True
                    # st.experimental_rerun()
            else:
                st.error("Passwords do not match!")
        else:
            st.error("All fields are required.")

# -------------------- Login --------------------

def login():
    st.title("🔑 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username and password:
            users = load_users()
            if username in users and check_password(password, users[username]):
                st.session_state['authenticated'] = True
                st.session_state['username'] = username
                st.success("Login Successful!")
                # st.experimental_rerun()
            else:
                st.error("Invalid credentials!")
        else:
            st.error("All fields are required.")

# -------------------- EEG Emotion Detection --------------------

def eeg_emotion_detection():
    st.title("🧠 EEG Emotion Detection")
    
    selected_features = [
        'stddev_2_a', 'min_2_a', 'min_q_2_a', 'min_q_7_a', 'min_q_12_a', 'min_q_17_a', 
        'covmat_104_a', 'logm_9_a', 'entropy0_a', 'entropy3_a', 'stddev_2_b', 'min_2_b',
        'min_q_2_b', 'min_q_7_b', 'min_q_12_b', 'min_q_17_b', 'covmat_104_b', 
        'logm_9_b', 'entropy0_b', 'entropy3_b'
    ]
    
    input_data = [st.number_input(f"{feature}", value=0.0, format="%.5f") for feature in selected_features]
    
    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = eeg_model.predict(scaled_input)
        predicted_class = np.argmax(prediction, axis=1)[0]
        emotion = label_encoder.inverse_transform([predicted_class])[0]
        st.success(f"Predicted Emotion: **{emotion}**")

# -------------------- Face Emotion Detection --------------------

def face_emotion_detection():
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model.
    classifier = load_model('model_78.h5')
    classifier.load_weights("model_weights_78.h5")

    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def predict_emotion(image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return image, "No face detected"
        
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = img_gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            
            prediction = classifier.predict(roi)[0]
            maxindex = int(np.argmax(prediction))
            finalout = emotion_labels[maxindex]
            output = str(finalout)
            
            label_position = (x, y - 10)
            cv2.putText(image, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return image, output

    # Streamlit App
    st.title("Image-based Face Emotion Detection 😀")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        processed_image, prediction = predict_emotion(image.copy())
        
        st.image(processed_image, caption="Processed Image with Prediction", use_column_width=True)
        st.write(f"Predicted Emotion: **{prediction}**")

# -------------------- Speech Emotion Detection --------------------

def speech_emotion_detection():
    st.title("🎙️ Speech Emotion Detection")
    
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        audio, rate = librosa.load(uploaded_file, sr=16000)
        inputs = feature_extractor(audio, sampling_rate=rate, return_tensors="pt", padding=True)
        outputs = speech_model(inputs.input_values)
        predictions = torch.nn.functional.softmax(outputs.logits.mean(dim=1), dim=-1)  # Average over sequence length
        predicted_label = torch.argmax(predictions, dim=-1)
        emotion = speech_model.config.id2label[predicted_label.item()]
        st.success(f"Predicted Emotion: **{emotion}**")

# -------------------- Main Page --------------------

if 'registered' not in st.session_state:
    signup()
elif 'authenticated' not in st.session_state:
    login()
else:
    st.sidebar.title("Menu")
    choice = st.sidebar.radio("Choose Option", ["EEG Emotion", "Face Emotion", "Speech Emotion"])
    
    if choice == "EEG Emotion":
        eeg_emotion_detection()
    elif choice == "Face Emotion":
        face_emotion_detection()
    elif choice == "Speech Emotion":
        speech_emotion_detection()
