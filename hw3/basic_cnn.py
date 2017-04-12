#!/usr/local/bin/python3
import sys
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten

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

    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(Conv2D(128, (2, 2), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X, Y,
              epochs=25,
              batch_size=128,
              validation_split=0.2)

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
