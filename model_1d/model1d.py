from keras import backend as K
from keras.constraints import maxnorm
from keras.layers import Concatenate
from keras.layers import Input
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model

def dense_block(x1):

    x2_temp = Dense(128,activation="relu",
                    kernel_initializer="normal",
                    kernel_constraint=maxnorm(3))(x1)
    x2_temp = Dropout(0.1)(x2_temp)
    x2 = Concatenate(axis=1)([x2_temp, x1])

    x3_temp = Dense(128,activation="relu",
                    kernel_initializer="normal",
                    kernel_constraint=maxnorm(3))(x2)
    x3_temp = Dropout(0.1)(x3_temp)
    x3 = Concatenate(axis=1)([x3_temp, x2, x1])

    x4_temp = Dense(128,activation="relu",
                    kernel_initializer="normal",
                    kernel_constraint=maxnorm(3))(x3)
    x4_temp = Dropout(0.1)(x4_temp)
    x4 = Concatenate(axis=1)([x4_temp, x3, x2, x1])

    return x4


def transition_block(x):

    x = Dense(128,activation="relu",
              kernel_initializer="normal",
              kernel_constraint=maxnorm(3))(x)
    x = Dropout(0.1)(x)

    return x


def transition_block_end(x):

    x = Dense(7,activation="relu",
              kernel_initializer="normal")(x)

    return x


def dense_1d(input_shape):

    img_input = Input(shape=input_shape)

    x = Dense(128,
              input_shape=input_shape,
              activation="relu",
              kernel_constraint=maxnorm(3))(img_input)
    x = Dropout(0.1)(x)
    
    x = dense_block(x)
    x = transition_block(x)
    x = dense_block(x)
    x = transition_block_end(x)
    
    model = Model(inputs=[img_input], outputs=[x])

    return model 

def fc_1d(input_shape):
    model = Sequential()

    model.add(Dense(128,input_shape=input_shape,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(7,activation="relu"))
    
    return model

def main():
    input_shape = (64,)

    model = dense_1d(input_shape)

    model.summary()

    plot_model(model,
               to_file="1dnet_im.png",
               show_shapes=True)
    

if __name__ == "__main__":
    main()
