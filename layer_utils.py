# Custom Keras layers and utility functions for the DeblurGAN generator
# Includes InstanceNormalization, ReflectionPadding2D, and residual blocks
import os
from keras.layers import Layer, Conv2D, Add, Activation, Dropout
import tensorflow as tf
from tensorflow.keras.layers import Layer
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings("ignore")

# Instance Normalization layer - normalizes each sample independently
# Used instead of Batch Normalization for style transfer and image generation tasks
class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    # Create learnable scale (gamma) and shift (beta) parameters
    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1],),
                                     initializer='ones',
                                     trainable=True,
                                     name='gamma')
        self.beta = self.add_weight(shape=(input_shape[-1],),
                                    initializer='zeros',
                                    trainable=True,
                                    name='beta')
        super(InstanceNormalization, self).build(input_shape)

    # Normalize inputs per-instance across spatial dimensions
    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    # Serialize layer config for model saving and loading
    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({
            "epsilon": self.epsilon,
        })
        return config

# Reflection padding layer to reduce boundary artifacts in convolutions
class ReflectionPadding2D(Layer):
    """Reflection Padding Layer (custom implementation)"""
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    # Compute output shape after reflection padding is applied
    def compute_output_shape(self, s):
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        return tf.pad(x, [[0, 0], [self.padding[0], self.padding[0]], [self.padding[1], self.padding[1]], [0, 0]], mode='REFLECT')

# Residual block with two conv layers, instance norm, and optional dropout
def res_block(x, filters, use_dropout=False):
    """Residual block with Instance Normalization."""
    res = ReflectionPadding2D((1, 1))(x)
    res = Conv2D(filters, (3, 3), padding='valid')(res)
    res = InstanceNormalization()(res)
    res = Activation('relu')(res)
    
    res = ReflectionPadding2D((1, 1))(res)
    res = Conv2D(filters, (3, 3), padding='valid')(res)
    res = InstanceNormalization()(res)
    
    if use_dropout:
        res = Dropout(0.5)(res)
    
    return Add()([x, res])

