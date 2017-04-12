#!/usr/local/bin/python3
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

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

    X_valid, X_train = X[:4000], X[4000:]
    Y_valid, Y_train = Y[:4000], Y[4000:]

    datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    datagen.fit(X_train)

    validgen = ImageDataGenerator(
        rescale=1./255)

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=128),
        steps_per_epoch=400,
        validation_data=validgen.flow(X_valid, Y_valid, batch_size=128),
        validation_steps=4000//128,
        epochs=400,
        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=0, mode='auto')]
        )

    model.save_weights('model_weights.hdf5')

    return model

def predict(model, test_features):
    X = test_features / 255
    X.shape = (-1, 48, 48, 1)
    return model.predict_classes(X)

def main():
    labels, features = read_features(sys.argv[1])
    model = train_model(features, labels)
    ids, test_features = read_features(sys.argv[2])
    results = predict(model, test_features)

    with open(sys.argv[3], 'w') as output_file:
        outputs = ['id,label']
        for i, label in enumerate(results):
            outputs.append('%d,%d' % (i, label))
        outputs.append('')
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()
