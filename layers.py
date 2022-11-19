import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, regularizers, backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Activation, Conv2D, Input, GlobalAveragePooling2D, Concatenate, InputLayer, \
ReLU, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Softmax, Lambda, LeakyReLU, Reshape, \
DepthwiseConv2D, Multiply, Add, LayerNormalization, Conv2DTranspose

# pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q

from keras_cv_attention_models import efficientnet, convnext, swin_transformer_v2

def layer_norm(inputs, zero_gamma=False, epsilon=1e-5):
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    return LayerNormalization(axis=-1, epsilon=epsilon, gamma_initializer=gamma_initializer)(inputs)

class wBiFPNAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=(num_in,),
                                initializer=tf.keras.initializers.constant(1 / num_in),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
          'epsilon': self.epsilon
        })
        return config

def conv(inputs, filters, kernel_size=3, strides=(1, 1), padding='same', activation=None):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(inputs)
    return x

def bn_act(inputs, activation='swish'):
    x = BatchNormalization()(inputs)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def conv_bn_act(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation='swish'):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    return x

def self_attention(inputs, filters):
    x = conv_bn_act(inputs, filters, 1, activation=None)
    norm_x = tf.math.l2_normalize(x)
    x = tf.keras.activations.relu(x)
    x = conv_bn_act(x, filters, 1, activation=None)
    x = tf.keras.activations.softplus(x)
    x = x * norm_x
    return x

def mkn_conv(inputs, filters, kernel_sizes=[1,3,5], padding="same", activation='swish'):
    # multi kernel size conv
    list_f = []
    for kernel_size in kernel_sizes:
        f = Conv2D(filters=filters, 
                   kernel_size=kernel_size,  
                   padding=padding, 
                   )(inputs)
        f = layer_norm(f)
        f = Activation(activation)(f)
        list_f.append(f)
    list_f = Concatenate()(list_f)
    return list_f

def atrous_conv(inputs, filters, dilation_rates=[6,12,18], kernel_size=3, padding="same", activation='swish'):
    list_f = []
    for rate in dilation_rates:
        f = Conv2D(filters=filters, 
                   kernel_size=kernel_size,  
                   padding=padding, 
                   dilation_rate=rate
                   )(inputs)
        f = layer_norm(f)
        f = Activation(activation)(f)
        list_f.append(f)
    
    list_f = Concatenate()(list_f)
    return list_f

def mkn_atrous_block(inputs, do_dim, final_dim, kernel_sizes, dilation_rates):
    mkn_list = mkn_conv(inputs, do_dim, kernel_sizes)
    atrous_list = atrous_conv(inputs, do_dim, dilation_rates)
    x = Concatenate()([mkn_list, atrous_list])
    x = Conv2D(filters=final_dim, 
               kernel_size=1,  
               padding="same", 
               )(x)
    return x