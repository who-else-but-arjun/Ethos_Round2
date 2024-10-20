# SRCNN - Super-Resolution Convolutional Neural Network
# Implements single image super-resolution using a three-layer CNN
# Reference: Dong et al., "Image Super-Resolution Using Deep Convolutional Networks"
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array, array_to_img
import numpy as np
import cv2
import sys
sys.stdout.reconfigure(encoding='utf-8')
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Build the SRCNN model architecture and load pre-trained weights
def build_model():
    SRCNN = Sequential()
    
    # Patch extraction and representation layer (9x9 filters)
    SRCNN.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(None, None, 1)))
    
    # Non-linear mapping layer (3x3 filters)
    SRCNN.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    
    # Reconstruction layer - produces the final high-resolution output (5x5 filters)
    SRCNN.add(Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    # Compile with Adam optimizer and MSE loss, then load pre-trained weights
    adam = Adam(learning_rate=0.0001)
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    SRCNN.load_weights('Weights_for_SRCNN.h5')
    
    return SRCNN

# Preprocess input image: convert to grayscale and normalize to [0, 1]
def preprocess_image(image):
    # Remove batch dimension if present
    if len(image.shape) == 4 and image.shape[0] == 1:
        image = np.squeeze(image, axis=0)
    if image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image 
    
    image = img_to_array(gray_image).astype(np.float32) / 255.0
    if len(image.shape) == 2: 
        image = np.expand_dims(image, axis=-1) 

    img = np.expand_dims(image, axis=0) 
    return img

# Convert model output back to displayable image format (uint8, 0-255)
def postprocess_image(pred):
    pred = np.squeeze(pred)
    pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
    
    return pred

# Run the full SRCNN enhancement pipeline on an input image
def enhance_images(image, model=build_model()):
    processed_img = preprocess_image(image)
    pred = model.predict(processed_img)
    enhanced_img = postprocess_image(pred)
    
    return enhanced_img

# Entry point for standalone testing
if __name__ == "__main__":
    srcnn_model = build_model()
    input_image = cv2.imread('path_to_image')
    enhanced_image = enhance_images(input_image, srcnn_model)
    cv2.imshow('Enhanced Image', enhanced_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
