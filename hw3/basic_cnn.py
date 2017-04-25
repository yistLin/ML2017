#!/usr/local/bin/python3
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import ModelCheckpoint

def read_features(filename):
    col0 = []
    col1 = []
    with open(filename, 'r') as f:
        next(f)
        for line in f:
            fields = line.strip().split(',')
            col0.append(int(fields[0]))
            col1.append(list(map(int, fields[1].split())))

    return np.array(col0), np.array(col1)

def train_model(features, labels):
    X = features / 255
    X.shape = (-1, 48, 48, 1)
    Y = np.zeros(shape=(labels.shape[0], 7))
    Y[np.arange(labels.shape[0]), labels] = 1

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    checkpointer = ModelCheckpoint(
            filepath='basic-model-{val_acc:.4f}.hdf5',
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            mode='auto',
            period=1)

    model.fit(X, Y,
            epochs=25,
            batch_size=128,
            validation_split=0.2,
            callbacks=[checkpointer])

    return model

def main():
    labels, features = read_features(sys.argv[1])
    model = train_model(features, labels)

if __name__ == '__main__':
    main()
