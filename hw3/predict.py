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
    return model.predict_classes(X)

def main():
    model = load_model(sys.argv[2])
    ids, test_features = read_features(sys.argv[1])
    results = predict(model, test_features)

    with open(sys.argv[3], 'w') as output_file:
        outputs = ['id,label']
        for i, label in enumerate(results):
            outputs.append('%d,%d' % (i, label))
        outputs.append('')
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()
