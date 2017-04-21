#!/usr/local/bin/python3
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger

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
    X = features.reshape(-1, 48, 48, 1)
    Y = np.zeros(shape=(labels.shape[0], 7))
    Y[np.arange(labels.shape[0]), labels] = 1

    X_valid, X_train = X[:3000], X[3000:]
    Y_valid, Y_train = Y[:3000], Y[3000:]
    X_train_flip = X_train[:, :, ::-1, :]
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train), axis=0)

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2)
    datagen.fit(X_train)
    X_valid = X_valid / 255.

    model = Sequential()

    model.add(Flatten(input_shape=(48, 48, 1)))

    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(2048, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    csv_logger = CSVLogger('dnn-training.log', separator=',', append=False)

    model.summary()

    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=128),
        steps_per_epoch=5*len(X_train)//128,
        validation_data=(X_valid, Y_valid),
        epochs=2000,
        callbacks=[csv_logger])

def main():
    labels, features = read_features(sys.argv[1])
    train_model(features, labels)

if __name__ == '__main__':
    main()

