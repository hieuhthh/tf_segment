from layers import *

# https://towardsdatascience.com/review-dilated-convolution-semantic-segmentation-9d5a5bd768f5

# conv(3,3) with dilate d
# receptive fields: 4d - 1
# d     field
# 1     3
# 2     7
# 3     11
# 4     15
# 5     19
# 6     23
# 7     27
# 8     31
# 12    47
# 16    63
# 18    71

config_map = {
        'x2' : {
            'kernel_sizes' : [3, 7],
            'dilation_rates' : [8, 16]
        },
        'x4' : {
            'kernel_sizes' : [3, 7],
            'dilation_rates' : [6, 8]
        },
        'x8' : {
            'kernel_sizes' : [3, 7],
            'dilation_rates' : [4, 6]
        },
        'x16' : {
            'kernel_sizes' : [1, 3],
            'dilation_rates' : [2]
        },
        'x32' : {
            'kernel_sizes' : [1, 3],
            'dilation_rates' : [2]
        },
    }

def create_model(im_size, n_labels, config_map, do_dim, final_dim):
    backbone = efficientnet.EfficientNetV2S((im_size,im_size,3), 
                                             pretrained="imagenet21k",
                                             num_classes=0)

    backbone_layer_names = [
                'stack_0_block1_output',
                'stack_1_block3_output',
                'stack_2_block3_output',
                'stack_4_block8_output',
                'post_swish'
            ]

    backbone_layers = [backbone.get_layer(layer_name).output for layer_name in backbone_layer_names]

    extract_layers = []

    for idx, key in enumerate(list(config_map.keys())):
        backbone_map = backbone_layers[idx]
        f = mkn_atrous_block(backbone_map, do_dim, final_dim, config_map[key]['kernel_sizes'], config_map[key]['dilation_rates'])
        f = Dropout(0.2)(f)
        extract_layers.append(f)

    temp_outputs = []

    for out_i, extract_layer in enumerate(extract_layers):
        temp_out = Conv2D(filters=n_labels, 
                          kernel_size=1,  
                          padding="same",
                          activation='sigmoid',
                          name=f'output_x{2**(out_i+1)}'
                          )(extract_layer)
        temp_outputs.append(temp_out)

    upsample_layers = []

    for idx, extract_layer in enumerate(extract_layers):
        stride = 2**(idx+1)
        ups = Conv2DTranspose(final_dim, 
                              kernel_size=(4, 4), 
                              strides=(stride, stride), padding="same")(extract_layer)
        ups = layer_norm(ups)
        ups = Activation("swish")(ups)
        upsample_layers.append(ups)

    x = Concatenate()(upsample_layers)
    x = Dropout(0.4)(x)

    x = self_attention(x, final_dim)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=final_dim//2, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = layer_norm(x)
    x = Activation("swish")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=final_dim//2, 
               kernel_size=3,  
               padding="same",
               )(x)
    x = layer_norm(x)
    x = Activation("swish")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters=n_labels, 
               kernel_size=1,  
               padding="same",
               activation='sigmoid',
               name=f'output_x1'
               )(x)

    temp_outputs.insert(0, x)

    out_dict = {'x1':temp_outputs[0], 'x2':temp_outputs[1], 'x4':temp_outputs[2], 
                'x8':temp_outputs[3], 'x16':temp_outputs[4], 'x32':temp_outputs[5]}

    model = Model(backbone.input, out_dict)
    
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

    model = create_model(im_size, n_labels, config_map, do_dim, final_dim)

    model.summary()

    print(model.output)

    inp = tf.ones((1, im_size, im_size, 3))
    out = model.predict(inp)