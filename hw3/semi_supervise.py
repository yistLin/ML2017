#!/usr/local/bin/python3
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten, LeakyReLU, BatchNormalization
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
    # Reshape data
    X = features.reshape(-1, 48, 48, 1)
    Y = np.zeros(shape=(labels.shape[0], 7))
    Y[np.arange(labels.shape[0]), labels] = 1

    # Normalize training data between 0 and 1
    X = X / 255.

    # Separate training, validation, and non-labeled data
    X_valid, X_train = X[:3000], X[3000:]
    Y_valid, Y_train = Y[:3000], Y[3000:]
    X_lab, X_nolab = X_train[:-12000], X_train[-12000:]
    Y_lab = Y_train[:-12000]

    print('# of X_valid =', len(X_valid))
    print('# of X_lab   =', len(X_lab))
    print('# of X_nolab =', len(X_nolab))

    # Image data generator
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    # Model structure
    model = Sequential()

    model.add(Conv2D(48, (3, 3), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())
 
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same', kernel_initializer='glorot_normal'))
    model.add(LeakyReLU(alpha=1./20))
    model.add(BatchNormalization())

    model.add(AveragePooling2D(pool_size=(3, 3), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(1024, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax', kernel_initializer='glorot_normal'))
    model.compile(optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

    for phase in range(500):
        # Save checkpoint
        checkpointer = ModelCheckpoint(
                filepath='semi-model-ph%d.hdf5' % phase,
                monitor='val_acc',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
                mode='auto',
                period=1)

        # Save training process
        csv_logger = CSVLogger('semi-training-ph%d.log' % phase, separator=',', append=False)

        # Training
        model.fit_generator(
                datagen.flow(X_lab, Y_lab, batch_size=128),
                steps_per_epoch=2*len(X_lab)//128,
                validation_data=(X_valid, Y_valid),
                epochs=10,
                callbacks=[checkpointer, csv_logger])

        # Predicting
        print()
        print('Phase %d end' % phase)
        print('Predicting...')
        predictions = model.predict(X_nolab)
        confidences = np.max(predictions, axis=1)
        Y_newlab = np.argmax(predictions[np.where(confidences > 0.7)], axis=1)
        X_newlab = X_nolab[np.where(confidences > 0.7)]
        X_nolab = X_nolab[np.where(confidences <= 0.7)]
        X_lab = np.concatenate((X_lab, X_newlab))
        Y_lab = np.concatenate((Y_lab, Y_newlab))
        print('# of X_newlab =', len(X_newlab))
        print('# of X_nolab  =', len(X_nolab))

def main():
    labels, features = read_features(sys.argv[1])
    train_model(features, labels)

if __name__ == '__main__':
    main()
