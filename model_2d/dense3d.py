from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
from keras.models import Model
from keras.layers import Activation
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import Concatenate
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.utils import plot_model

def dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x, 64, name=name + '_block' + str(i + 1))
    return x


def transition_block(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(128, (1,1,1), use_bias=False,
               name=name + '_conv')(x)

    return x

def transition_block_end(x, reduction, name):
    """A transition block.
    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name=name + '_bn')(x)
    x = Activation('relu', name=name + '_relu')(x)
    x = Conv3D(1, (1,1,1), use_bias=False,
               name=name + '_conv')(x)

    return x

def conv_block(x, growth_rate, name):
    """A building block for a dense block.
    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    bn_axis = 4 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_0_bn')(x)
    x1 = Activation('relu', name=name + '_0_relu')(x1)
    x1 = Conv3D(2* growth_rate, (1,1,1), use_bias=False,
                name=name + '_1_conv')(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                            name=name + '_1_bn')(x1)
    x1 = Activation('relu', name=name + '_1_relu')(x1)
    x1 = Conv3D(growth_rate, (3,3,3), padding='same', use_bias=False,
                name=name + '_2_conv')(x1)
    x = Concatenate(axis=bn_axis, name=name + '_concat')([x, x1])
    return x

def dense_net(input_shape=None):
    """Instantiates the DenseNet architecture.
    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.
    The model and the weights are compatible with
    TensorFlow, Theano, and CNTK. The data format
    convention used by the model is the one
    specified in your Keras config file.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    blocks = [2, 2, 2, 2]
    
    img_input = Input(shape=input_shape)

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = Conv3D(64, (3,3,3), padding="same",
               use_bias=False, name="conv1/conv")(img_input)
    x = BatchNormalization(axis=bn_axis, epsilon=1.001e-5,
                           name="conv1/bn")(x)
    x = Activation("relu", name="conv1/relu")(x)

    depth = len(blocks)
    for ii in range(depth):
        x = dense_block(x, blocks[ii], name=("conv%i"  % (ii+2)))
        if ii+1 < depth:
            x = transition_block(x, 0.5, name=("pool%i" % (ii+2)))
        else:
            x = transition_block_end(x, 0.5, name=("pool%i" % (ii+2)))
            
    x = Activation("relu", name="relu")(x)
    
    model = Model(inputs=[img_input], outputs=[x])

    return model

def main():

    image_size = (128, 128, 128, 1)

    model = dense_net(image_size)

    model.summary()
    
    plot_model(model,
               to_file="3dnet_im.png",
               show_shapes=True)


if __name__ == "__main__":
    main()
