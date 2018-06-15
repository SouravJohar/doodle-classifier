from keras.models import Sequential
from keras.layers import Dense, Dropout


def mlp(classes):
    model = Sequential()
    model.add(Dense(units=600, activation='relu', input_dim=784))
    model.add(Dropout(0.3))
    model.add(Dense(units=400, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=100, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=25, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=classes, activation='softmax'))
    return model
