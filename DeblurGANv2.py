# DeblurGANv2 - Image deblurring using a Generative Adversarial Network (GAN)
# This module implements the generator architecture for deblurring images
import os
import numpy as np

import cv2
from keras.layers import Input, Activation, Add, UpSampling2D, Conv2D, Lambda
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array, array_to_img
import matplotlib.pyplot as plt
from layer_utils import ReflectionPadding2D, res_block
from layer_utils import InstanceNormalization 
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Model configuration parameters
channel_rate = 64
# Input image dimensions (height, width, channels)
image_shape = (256, 256, 3)
patch_shape = (channel_rate, channel_rate, 3)

# Generator and discriminator filter counts
ngf = 64
ndf = 64
# Number of input and output channels (RGB)
input_nc = 3
output_nc = 3
input_shape_generator = (256, 256, input_nc)
input_shape_discriminator = (256, 256, output_nc)
# Number of residual blocks in the generator network
n_blocks_gen = 18

def generator_model():
    """Build generator architecture."""
    inputs = Input(shape=image_shape)

    x = ReflectionPadding2D((3, 3))(inputs)
    x = Conv2D(filters=ngf, kernel_size=(7, 7), padding='valid')(x)
    x = InstanceNormalization()(x)
    x = Activation('relu')(x)

    # Downsampling layers to reduce spatial dimensions
    n_downsampling = 2
    for i in range(n_downsampling):
        mult = 2**i
        x = Conv2D(filters=ngf*mult*2, kernel_size=(3, 3), strides=2, padding='same')(x)
        x = InstanceNormalization()(x) 
        x = Activation('relu')(x)

    # Apply residual blocks for feature transformation
    mult = 2**n_downsampling
    for i in range(n_blocks_gen):
        x = res_block(x, ngf*mult, use_dropout=True)

    # Upsampling layers to restore spatial dimensions
    for i in range(n_downsampling):
        mult = 2**(n_downsampling - i)
        x = UpSampling2D()(x)
        x = Conv2D(filters=int(ngf * mult / 2), kernel_size=(3, 3), padding='same')(x)
        x = InstanceNormalization()(x) 
        x = Activation('relu')(x)

    # Final output layer with tanh activation to produce pixel values in [-1, 1]
    x = ReflectionPadding2D((3, 3))(x)
    x = Conv2D(filters=output_nc, kernel_size=(7, 7), padding='valid')(x)
    x = Activation('tanh')(x)
    outputs = x  

    model = Model(inputs=inputs, outputs=outputs, name='Generator')
    return model

def load_generator_weights(generator, weights_path):
    """Load pre-trained weights into the generator model."""
    generator.load_weights(weights_path, by_name=True, skip_mismatch=True)
    return generator

def preprocess_image(image, image_shape=(256, 256)):
    """Preprocess the image for the model."""
    img = cv2.resize(image, image_shape[:2])
    if len(img.shape) == 2: 
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

    image = img_to_array(img)
    image = (image / 127.5) - 1
    return np.expand_dims(image, axis=0)

def postprocess_image(deblurred_image):
    """Post-process the deblurred image from the model output."""
    deblurred_image = (deblurred_image + 1) * 127.5 
    deblurred_image = np.clip(deblurred_image, 0, 255).astype('uint8')
    return array_to_img(deblurred_image[0])

def deblur_image(image, weights_path='Weights_for_DeblurGANv2.h5'):
    """Deblur all images in the input folder using the generator model."""
    generator = generator_model()
    generator = load_generator_weights(generator, weights_path)
    image = preprocess_image(image)
    deblurred_image = generator.predict(image)
    result_image = postprocess_image(deblurred_image)
    return result_image

# Entry point for standalone execution
if __name__ == '__main__':
    input_folder = 'image'      
    output_folder = 'image_out'  
    weights_path = 'Weights_for_DeblurGANv2.h5'

    deblur_image(input_folder, weights_path)
