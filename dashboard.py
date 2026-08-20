import os
import streamlit as st
import pandas as pd
import tempfile
import time
import cv2
import numpy as np
from keras.models import load_model
from pipeline import main
import json
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

model = load_model('Trained_on_dataset.h5')
with open('Class.json') as f:
    class_data = json.load(f)

st.set_page_config(page_title="Facial Reconstruction Dashboard", layout="wide")
st.title("Facial Reconstruction Dashboard")

def get_suspect_details():
    return {
        "Name": "Arjun Verma", 
        "ID": "230102125", 
        "Status": "ECE"
    }

def generate_logs():
    return f"[{time.strftime('%H:%M:%S')}] Processing video..."

def preprocess_image(image):  
    image = cv2.resize(image, (224, 224))  
    if len(image.shape) == 2: 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0 
    return image

def process_video(video_path, suspect_name):
    found_faces = []
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            processed_face = preprocess_image(face)
            final = main(processed_face)
            if isinstance(final, np.ndarray):
                final = final 
            else:
                final = np.array(final) 
            if final.ndim == 2: 
                final = np.expand_dims(final, axis=-1)
            final = cv2.cvtColor(final, cv2.COLOR_RGB2BGR)
            final = cv2.resize(final, (224, 224))
            final = np.expand_dims(final, axis=0)

            prediction = model.predict(final)
            confidence = np.max(prediction)
            class_id = np.argmax(prediction)

            suspect_info = get_suspect_details()
            predicted_class_info = class_data.get(str(class_id), {"Name": "Unknown", "ID": "N/A", "Status": "N/A"})
            if predicted_class_info["Name"] == suspect_info["Name"]:
                found_faces.append((face, class_id, confidence))
                cap.release() 
                return found_faces

    cap.release()
    return None 

left_col, right_col = st.columns(2)
found_face_info = None

with left_col:
    st.header("Upload Pre-Recorded Video")
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_video_path = temp_file.name

        st.subheader("Video Playback")
        st.video(temp_video_path)
        st.subheader("Terminal Output")
        if st.button("Show Logs"):
            logs = generate_logs()
            st.text_area("Logs", logs, height=200)

        suspect_details = get_suspect_details()
        suspect_name = suspect_details["Name"]
        suspect_image_path = f"Headsets/{suspect_name}/1.jpg"
        suspect_image = cv2.imread(suspect_image_path)
        if suspect_image is None:
            st.error(f"Error loading image for suspect: {suspect_name}. Please check the image path: {suspect_image_path}")
        else:
            found_face_info = process_video(temp_video_path, suspect_name)

with right_col:
    st.header("Suspect Details")
    suspect_info = get_suspect_details()
    
    suspect_df = pd.DataFrame(suspect_info.items(), columns=["Field", "Value"])
    st.subheader(f"Details for {suspect_info['Name']}")
    st.table(suspect_df)

    if found_face_info:
        found_face, class_id, confidence = found_face_info[0]
        found_face = cv2.cvtColor(found_face, cv2.COLOR_BGR2RGB)

        predicted_class_info = class_data.get(str(class_id), {"Name": "Unknown", "ID": "N/A", "Status": "N/A"})
        st.write(f"Predicted Name: {predicted_class_info['Name']}")
        st.write(f"Predicted ID: {predicted_class_info['ID']}")
        st.write(f"Predicted Status: {predicted_class_info['Status']}")

        st.header("Comparison of Found and Suspect Images")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Found Image")
            st.image(found_face, caption=f"Image from Video (Confidence: {confidence:.2f})", channels="RGB", width=200)

        with col2:
            suspect_face_rgb = cv2.cvtColor(suspect_image, cv2.COLOR_BGR2RGB)  
            st.subheader("Suspect Image")
            st.image(suspect_face_rgb, caption="Actual Suspect", channels="RGB", width=200)
    else:
        st.write("No faces found in the video.")
