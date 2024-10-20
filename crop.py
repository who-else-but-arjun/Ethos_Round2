# Face cropping utility for dataset preprocessing
# Detects faces using Haar cascade and crops them to a uniform size
import cv2
import os
from pathlib import Path

# Load the Haar cascade classifier for frontal face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Iterate through dataset subfolders, detect faces, and save cropped images
def crop(input_folder, output_folder):
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        
        if os.path.isdir(subfolder_path):
            output_subfolder_path = os.path.join(output_folder, subfolder)
            if not os.path.exists(output_subfolder_path):
                os.makedirs(output_subfolder_path)
            image_count = 1
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_file)

                image = cv2.imread(image_path)
                if image is None:
                    continue
                
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Detect faces with multi-scale detection parameters
                faces = face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=8, 
                    minSize=(50, 50)
                )
                for (x, y, w, h) in faces:
                    # Filter faces by aspect ratio to avoid false positives
                    aspect_ratio = w / float(h)
                    if 0.8 < aspect_ratio < 1.2:
                        face = image[y:y+h, x:x+w]
                        # Resize cropped face to 224x224 for model input
                        face_resized = cv2.resize(face, (224, 224))
                        output_image_path = os.path.join(output_subfolder_path, f"{image_count}.jpg")
                        cv2.imwrite(output_image_path, face_resized)
                        print(f"Saved: {output_image_path}")
                        
                        image_count += 1

# Entry point for standalone execution
if __name__ == "__main__":
    input_folder = 'Dataset'
    output_folder = 'Headsets'
    
    crop(input_folder, output_folder)
