from layers import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet, convnext, swin_transformer_v2

def create_model(im_size, n_labels, use_dim):
    backbone = swin_transformer_v2.SwinTransformerV2Tiny_window16((im_size,im_size,3), 
                                                                  pretrained="imagenet",
                                                                  num_classes=0)

    backbone_layer_names = [
                'stack1_block2_output',
                'stack2_block2_output',
                'stack3_block2_output',
                'stack4_block2_output'
            ]

    backbone_layers = [backbone.get_layer(layer_name).output for layer_name in backbone_layer_names]

    extract_layers = []

    for backbone_layer in backbone_layers:
        astr = atrous_conv(backbone_layer, use_dim)
        astr = Dropout(0.1)(astr)
        extract_layers.append(astr)

    upsample_layers = []

    first_layer = extract_layers[0]
    first_layer = Conv2D(filters=use_dim, 
                         kernel_size=3,  
                         padding="same",
                         )(first_layer)
    first_layer = layer_norm(first_layer)
    first_layer = Activation("swish")(first_layer)
    upsample_layers.append(first_layer)

    for idx, extract_layer in enumerate(extract_layers[1:]):
        stride = 2**(idx+1)
        ups = Conv2DTranspose(use_dim, 
                              kernel_size=(3, 3), 
                              strides=(stride, stride), padding="same")(extract_layer)
        ups = layer_norm(ups)
        ups = Activation("swish")(ups)
        upsample_layers.append(ups)

    x = wBiFPNAdd()(upsample_layers)
    x = Dropout(0.3)(x)

    x = self_attention(x, use_dim)

    x = Conv2D(filters=use_dim, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = layer_norm(x)
    x = Activation("swish")(x)
    x = Dropout(0.1)(x)

    x = Conv2DTranspose(use_dim // 2, kernel_size=(3, 3), strides=(4, 4), padding="same")(x)
    x = layer_norm(x)
    x = Activation("swish")(x)
    x = Dropout(0.1)(x)
                        
    x = Conv2D(filters=use_dim // 2, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = layer_norm(x)
    x = Activation("swish")(x)
    x = Dropout(0.1)(x)
    
    x = Conv2D(filters=n_labels, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = Activation("sigmoid")(x)

    model = Model(backbone.input, x)
    
    return model

if __name__ == "__main__":
    import os
    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    n_labels = 1

    model = create_model(im_size, n_labels, use_dim)

    model.summary()

    inp = tf.ones((1, im_size, im_size, 3))
    out = model.predict(inp)