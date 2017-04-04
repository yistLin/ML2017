#!/usr/local/bin/python3
import sys
import numpy as np

def read_provided_features(x_filename, y_filename):
    features = []
    with open(x_filename, 'r') as x_file:
        next(x_file) # skip the first row
        for line in x_file:
            fields = line.strip().split(',')
            features.append(list(map(float, fields)))

    labels = []
    with open(y_filename, 'r') as y_file:
        for line in y_file:
            label = float(line.strip())
            labels.append(label)

    return np.array(features), np.array(labels)

def read_test_features(test_filename):
    tests = []
    with open(test_filename, 'r') as test_file:
        next(test_file) # skip the first row
        for line in test_file:
            fields = line.strip().split(',')
            tests.append(list(map(float, fields)))

    return np.array(tests)

def sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 0.00000000000001, 0.99999999999999)

def maximum_likelihood(X, Y, X_test):
    # split classes
    nb_data, nb_feat = X.shape
    cx = [X[Y == 0], X[Y == 1]]
    p = [float(len(cx[0])) / nb_data, float(len(cx[1])) / nb_data]

    # calculate covariance matrix
    mu = [np.mean(cx[0], axis=0), np.mean(cx[1], axis=0)]
    sigma = [np.zeros(shape=(nb_feat,nb_feat)), np.zeros(shape=(nb_feat,nb_feat))]
    for i in range(2):
        for n in range(len(cx[i])):
            dif = (cx[i][n] - mu[i]).reshape(-1, 1)
            sigma[i] += np.dot(dif, dif.T)
        sigma[i] /= float(len(cx[i]))

    # shared covariance matrix
    shared_sigma = p[0] * sigma[0] + p[1] * sigma[1]

    # predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot((mu[1] - mu[0]), sigma_inverse)
    b = (-0.5) * np.dot(np.dot([mu[1]], sigma_inverse), mu[1]) + (0.5) * np.dot(np.dot([mu[0]], sigma_inverse), mu[0]) + np.log(float(len(cx[1]))/len(cx[0]))
    h = np.dot(w, X_test.T) + b

    return np.around(sigmoid(h))

def main():
    # features.shape = (32561, 106)
    features, labels = read_provided_features(sys.argv[3], sys.argv[4])

    # test.shape = (16281, 106)
    tests = read_test_features(sys.argv[5])

    predictions = maximum_likelihood(features, labels, tests)

    with open(sys.argv[6], 'w') as output_file:
        outputs = ['id,label']
        for i, label in enumerate(predictions):
            outputs.append('%d,%d' % (i+1, label))
        outputs.append('')
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()
