from layers import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet, convnext, swin_transformer_v2

def create_model(im_size, n_labels, use_dim, max_frames,
                 mlp_dim, num_heads, trans_layers, mha_dropout):
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
        extract_layers += astr

    x = tf.stack(extract_layers, 1)

    pe = PositionEmbedding(input_shape=(max_frames, use_dim),
                           input_dim=max_frames,
                           output_dim=use_dim,
                           mode=PositionEmbedding.MODE_ADD,
    )

    x = pe(x)

    for i in range(trans_layers):
        x = TransformerEncoder(use_dim, mlp_dim, num_heads)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = Dropout(mha_dropout)(x)

    encoder_map = backbone_layers[0]
    encoder_map = Conv2D(filters=use_dim, 
                         kernel_size=3,  
                         padding="same",
                         )(encoder_map)
    encoder_map = layer_norm(encoder_map)
    encoder_map = Activation("swish")(encoder_map)

    x = Reshape((1, 1, -1))(x)
    x = tf.math.multiply(encoder_map, x)

    x = Conv2D(filters=use_dim, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = layer_norm(x)
    x = Activation("swish")(x)

    x = Conv2DTranspose(use_dim, kernel_size=(3, 3), strides=(4, 4), padding="same")(x)
    x = layer_norm(x)
    x = Activation("swish")(x)

    x = Conv2D(filters=use_dim, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = layer_norm(x)
    x = Activation("swish")(x)

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

    model = create_model(im_size, n_labels, use_dim, max_frames,
                         mlp_dim, num_heads, trans_layers, mha_dropout)

    model.summary()

    inp = tf.ones((1, im_size, im_size, 3))
    out = model.predict(inp)