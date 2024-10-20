# VGGFace - Face recognition model based on the VGG16 architecture
# Uses pre-trained VGGFace weights for feature extraction
# Includes classification layers for fine-tuning on custom datasets
import os
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D, Flatten, Dense, Input
from keras.models import Model
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Image dimensions and training configuration
IMG_HEIGHT = 224  
IMG_WIDTH = 224
BATCH_SIZE = 32
# Number of identity classes for face recognition
NO_CLASSES = 10 

# Build the VGG16 base model and load pre-trained VGGFace weights
def vgg16(input_shape=(224, 224, 3)):
    inputs = Input(shape=input_shape)
    
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #x = Flatten()(x)
    
    # Create model and load pre-trained VGGFace weights
    model = Model(inputs, x, name="VGG16")
    model.load_weights('Weights_for_VGGFace.h5', by_name=True)
    model.summary()

    return model

# Add classification head on top of the VGG16 base for fine-tuning
def classification_layers(base):
    NO_CLASSES = 10
    x = base.output
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(base.output)

    # Fully connected layers for classification
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)

    preds = Dense(NO_CLASSES, activation='softmax')(x)
    model = Model(inputs = base.input, outputs = preds)
    
    # Freeze early convolutional layers to preserve learned features
    for layer in model.layers[:20]:
        layer.trainable = False

    # Allow later layers to be fine-tuned on the custom dataset
    for layer in model.layers[20:]:
        layer.trainable = True
        
    return model


        


