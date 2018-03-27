from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.utils import plot_model

def fc_1d(input_shape):
    model = Sequential()

    model.add(Dense(128,input_shape=input_shape,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(256,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(512,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(3,activation="relu"))
    
    return model

def main():
    input_shape = (64,)

    model = fc_1d(input_shape)

    model.summary()

    plot_model(model,
               to_file="1dnet_im.png",
               show_shapes=True)
    

if __name__ == "__main__":
    main()
