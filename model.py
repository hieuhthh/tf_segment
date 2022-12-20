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

def create_model(im_size, n_labels, final_dim, drop_block=0):
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

    modify_backbone = []

    for lay in backbone_layers:
        x = convnext_block(lay, lay.shape[-1], drop_rate=drop_block)
        modify_backbone.append(x)

    mixed_layers = []

    mix_inputs = modify_backbone

    for i in range(len(mix_inputs)):
        temp_mix = [mix_inputs[i]]

        for j in range(len(mix_inputs)):
            if i == j:
                continue
            scale = 2 ** (j - i)
            temp_up = upsample_resize(mix_inputs[j], scale)

            temp_mix.append(temp_up)

        x = concat_self_attn(temp_mix, mix_inputs[i].shape[-1])

        mixed_layers.append(x)

    modify_mixs = []

    for lay in mixed_layers:
        x = convnext_block(lay, lay.shape[-1])
        modify_mixs.append(x)

    extract_layers = []

    temp_outputs = []

    for idx, x in enumerate(modify_mixs):
        temp_out = Conv2D(filters=n_labels, 
                          kernel_size=1,  
                          padding="same",
                          activation='sigmoid',
                          name=f'output_{idx+1}'
                          )(x)
        temp_outputs.append(temp_out)

        x = upsample_resize(x, scale=2**(idx + 2))
        extract_layers.append(x)

    x = concat_self_attn(extract_layers, final_dim)
    x = convnext_block(x, final_dim)

    x = Conv2D(filters=n_labels, 
               kernel_size=1,  
               padding="same",
               activation='sigmoid',
               name=f'output_0'
               )(x)

    out_dict = {'x1':x, 'x2':temp_outputs[0], 'x3':temp_outputs[1], 
                'x4':temp_outputs[2], 'x5':temp_outputs[3]}

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

    model = create_model(im_size, n_labels, final_dim, drop_block)

    model.summary()

    print(model.output)

    inp = tf.ones((1, im_size, im_size, 3))
    out = model.predict(inp)