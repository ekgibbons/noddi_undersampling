from keras.models import Sequential
from keras.layers.core import Dense, Dropout

def fc_1d(input_shape):
    model = Sequential()

    model.add(Dense(150,input_shape=input_shape,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(150,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(150,activation="relu"))
    model.add(Dropout(0.1))

    model.add(Dense(1,activation="linear"))
    
    return model

def main():
    input_shape = (64,)

    model = fc_1d(input_shape)

    model.summary()
    

if __name__ == "__main__":
    main()
