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

def layer_norm(inputs, zero_gamma=False, epsilon=1e-6):
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
    x = Conv2D(filters, kernel_size=1, strides=(1,1), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    norm_x = tf.math.l2_normalize(x)
    x = tf.keras.activations.relu(x)
    x = Conv2D(filters, kernel_size=1, strides=(1,1), padding='same')(x)
    x = tf.keras.activations.softplus(x) 
    x = x * norm_x
    return x

def mkn_conv(inputs, filters, kernel_sizes=[1,3,5], padding="same", activation='swish', do_concat=False, dropout=0):
    # multi kernel size conv
    list_f = []
    for kernel_size in kernel_sizes:
        f = Conv2D(filters=filters, 
                   kernel_size=kernel_size,  
                   padding=padding, 
                   )(inputs)
        f = bn_act(f)
        
        if dropout > 0:
            f = Dropout(dropout)(f)
            
        list_f.append(f)
        
    if do_concat:
        list_f = Concatenate()(list_f)
        
    return list_f

def atrous_conv(inputs, filters, dilation_rates=[6,12,18], kernel_size=3, padding="same", activation='swish', do_concat=False, dropout=0):
    list_f = []
    for rate in dilation_rates:
        f = Conv2D(filters=filters, 
                   kernel_size=kernel_size,  
                   padding=padding, 
                   dilation_rate=rate
                   )(inputs)
        f = bn_act(f)
        
        if dropout > 0:
            f = Dropout(dropout)(f)
            
        list_f.append(f)
    
    if do_concat:
        list_f = Concatenate()(list_f)
        
    return list_f

def mkn_atrous_block(inputs, do_dim, kernel_sizes=None, dilation_rates=None, dropout=0, do_concat=False):
    f = []
    
    if kernel_sizes is not None:
        mkn_list = mkn_conv(inputs, do_dim, kernel_sizes, dropout=dropout)
        f += mkn_list
        
    if dilation_rates is not None:
        atrous_list = atrous_conv(inputs, do_dim, dilation_rates, dropout=dropout)
        f += atrous_list
        
    if do_concat:
        f = Concatenate()(f)
        
    return f

class softmax_merge(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(softmax_merge, self).__init__(**kwargs)

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=(num_in,),
                                initializer=tf.keras.initializers.constant(1 / num_in),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.softmax(tf.expand_dims(self.w, 0))[0]
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(softmax_merge, self).get_config()
        return config
    
def upsample_conv(inputs, scale=2, filters=None):
    if scale == 1:
        return inputs

    if filters is None:
        filters = inputs.shape[-1]

    if scale > 1:
        s = Conv2DTranspose(filters, 
                            kernel_size=(2, 2), 
                            strides=(scale, scale), 
                            padding="same")(inputs)
    else:
        scale = int(1.0 / scale)
        s = Conv2D(filters, 
                   kernel_size=(2, 2), 
                   strides=(scale, scale), 
                   padding="same")(inputs)

    s = bn_act(s)
    
    return s

def upsample_resize(inputs, scale=2):
    if scale == 1:
        return inputs

    ups = tf.image.resize(inputs, (int(inputs.shape[1]*scale), 
                                   int(inputs.shape[2]*scale)))
    return ups

def upsample_new(inputs, scale=2):
    if scale == 1:
        return inputs

    up_res = upsample_resize(inputs, scale)
    up_conv = upsample_conv(inputs, scale)

    return up_res + up_conv

def mlp(inputs, filters, activation='gelu', dropout=0, n_do=2):
    x = Dense(filters, activation)(inputs)
    x = Dropout(dropout)(x)
    for _ in range(n_do-1):
        x = Dense(filters, activation)(x)
        x = Dropout(dropout)(x)
    return x

def concat_merge(inputs, filters):
    """
    inputs: [l1, l2, ...]
    """
    x = Concatenate()(inputs)
    x = conv_bn_act(x, filters, 1)
    return x

def concat_self_attn(inputs, filters):
    """
    inputs: [l1, l2, ...]
    """
    x = Concatenate()(inputs)
    x = self_attention(x, filters)
    return x

class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config

def convnext_block(inputs, filters, layer_scale_init_value=1e-6, drop_rate=0, activation="gelu"):
    x = DepthwiseConv2D(kernel_size=7, padding="SAME", use_bias=True)(inputs)
    x = layer_norm(x)
    x = Dense(4 * filters)(x)
    x = Activation(activation=activation)(x)
    x = Dense(filters)(x)
    if layer_scale_init_value > 0:
        x = ChannelAffine(use_bias=False, weight_init_value=layer_scale_init_value)(x)
    x = Dropout(drop_rate)(x)
    return Add()([inputs, x])