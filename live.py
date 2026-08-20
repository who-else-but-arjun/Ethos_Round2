import streamlit as st
import cv2
import time
import numpy as np
from keras.models import load_model
from pipeline import main 
import json
import pandas as pd

model = load_model('Trained_on_dataset.h5')
with open('Class.json') as f:
    class_data = json.load(f)

st.set_page_config(page_title="Facial Reconstruction Dashboard", layout="wide")
st.title("Facial Reconstruction Dashboard")

if 'streaming' not in st.session_state:
    st.session_state.streaming = False
if 'found_face_info' not in st.session_state:
    st.session_state.found_face_info = None

def get_suspect_details():
    return {
        "Name": "Arjun Verma",
        "ID": "230102125",
        "Status": "ECE",
    }

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1: 
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image / 255.0
    return image

def video_stream():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Cannot access the webcam.")
        return

    frame_placeholder = st.empty()

    while st.session_state.streaming:
        ret, frame = cap.read()
        if not ret:
            st.warning("No frame received.")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5)
        face_detected = False

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
            predicted_class_info = class_data.get(str(class_id), {"Name": "Unknown", "ID": "N/A", "Status": "N/A"})
            face_detected = True

            st.session_state.found_face_info = (face, class_id, confidence)

            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)

        frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # If face detected, break the loop
        if face_detected:
            break

        time.sleep(0.03)

    cap.release()

left_col, right_col = st.columns(2)

with left_col:
    st.header("Live Webcam Stream")
    if not st.session_state.streaming:
        if st.button("Start Webcam Stream", key="start_button"):
            st.session_state.streaming = True
            video_stream()
    else:
        if st.button("Stop Stream", key="stop_button"):
            st.session_state.streaming = False

    if not st.session_state.streaming:
        st.write("Press 'Start Webcam Stream' to begin.")

    st.subheader("Terminal Output")
    if st.button("Show Logs", key="logs_button"):
        st.text_area("Logs", "[INFO] Streaming from webcam...", height=200)

with right_col:
    st.header("Suspect Details")
    suspect_info = get_suspect_details()
    suspect_df = pd.DataFrame(suspect_info.items(), columns=["Field", "Value"])
    st.table(suspect_df)

    if st.session_state.found_face_info:
        found_face, class_id, confidence = st.session_state.found_face_info
        found_face_rgb = cv2.cvtColor(found_face, cv2.COLOR_BGR2RGB)

        predicted_class_info = class_data.get(str(class_id), {"Name": "Unknown", "ID": "N/A", "Status": "N/A"})
        st.write(f"Name: {predicted_class_info['Name']}")
        st.write(f"ID: {predicted_class_info['ID']}")
        st.write(f"Status: {predicted_class_info['Status']}")

        if predicted_class_info["Name"] == suspect_info["Name"]:
            st.header("Match Found!")
        
        st.header("Comparison of Found and Suspect Images")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Found Image")
            st.image(found_face_rgb, caption=f"Image from Video (Confidence: {confidence:.2f})", channels="RGB", width=200)

        with col2:
            suspect_image_path = f"Headsets/{suspect_info['Name']}/1.jpg"
            suspect_face = cv2.imread(suspect_image_path)
            if suspect_face is not None:
                suspect_face_rgb = cv2.cvtColor(suspect_face, cv2.COLOR_BGR2RGB)  
                st.subheader("Suspect Image")
                st.image(suspect_face_rgb, caption="Actual Suspect", channels="RGB", width=200)
            else:
                st.error("Error loading suspect image.")
