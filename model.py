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

def create_model(im_size, n_labels, do_dim, kernel_sizes, dilation_rates, drop_block):
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

    temp_outputs = []

    for idx, backbone_layer in enumerate(backbone_layers):
        x = mkn_atrous_block(backbone_layer, do_dim, kernel_sizes, dilation_rates, drop_block)
        x = softmax_merge()(x)
        x = self_attention(x, do_dim)
        x = Dropout(drop_block)(x)

        temp_out = Conv2D(filters=n_labels, 
                          kernel_size=1,  
                          padding="same",
                          activation='sigmoid',
                          name=f'output_{idx+1}'
                          )(x)
        temp_outputs.append(temp_out)

        x = upsample(x, do_dim, scale=4*(2**idx))
        extract_layers.append(x)

    x = softmax_merge()(extract_layers)
    x = self_attention(x, do_dim)
    x = Dropout(drop_block)(x)
    x = mlp(x, do_dim // 2, 'gelu', drop_block)

    x = Conv2D(filters=n_labels, 
               kernel_size=1,  
               padding="same",
               activation='sigmoid',
               name=f'output_0'
               )(x)

    out_dict = {'x0':x, 'x1':temp_outputs[0], 'x2':temp_outputs[1], 
                'x3':temp_outputs[2], 'x4':temp_outputs[3]}

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

    model = create_model(im_size, n_labels, do_dim, kernel_sizes, dilation_rates, drop_block)

    model.summary()

    print(model.output)

    inp = tf.ones((1, im_size, im_size, 3))
    out = model.predict(inp)