#!/usr/local/bin/python3
import sys
import numpy as np
from keras.models import load_model

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

def predict(model, test_features):
    X = test_features / 255
    X.shape = (-1, 48, 48, 1)
    return model.predict(X)

def main():
    ids, test_features = read_features(sys.argv[1])
    probs = np.zeros(shape=(test_features.shape[0], 7))
    for model_file in sys.argv[3:]:
        model = load_model(model_file)
        probs += predict(model, test_features)

    results = np.argmax(probs, axis=-1)

    with open(sys.argv[2], 'w') as output_file:
        outputs = ['id,label']
        for i, label in enumerate(results):
            outputs.append('%d,%d' % (i, label))
        outputs.append('')
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()

