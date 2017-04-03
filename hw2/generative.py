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

def maximum_likelihood(X, Y, X_test):
    # normalization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # split classes
    cx = [X[Y == 0], X[Y == 1]]
    mu = []
    sigma = []

    # calculate mu and covariance matrix
    nb_data, nb_feat = X.shape
    for i in range(2):
        mu.append(np.mean(cx[i], axis=0))
        cov_sum = np.zeros(shape=(nb_feat, nb_feat))
        for n in range(len(cx[i])):
            dif = cx[i][n] - mu[i]
            dif = dif.reshape(-1, 1)
            cov_sum += (dif * dif.T)
        sigma.append(cov_sum / len(cx[i]))
        print('sigma[%d] =' % i, sigma[i])

    predictions = []
    term1 = []
    print('np.linalg.det(sigma[0]) =', np.linalg.det(sigma[0]))
    print('np.linalg.det(sigma[1]) =', np.linalg.det(sigma[1]))
    term1.append(1 / np.sqrt(2 * np.pi * np.linalg.det(sigma[0])))
    term1.append(1 / np.sqrt(2 * np.pi * np.linalg.det(sigma[1])))
    for i in range(len(X_test)):
        dif = X_test[i] - mu[0]
        dif = dif.reshape(-1, 1)
        term2 = np.exp(-0.5 * dif.T * (1/sigma[0]) * dif)
        print('P(C1|x): %f x %f = %f' % (term1[0], term2[0][0], term1[0] * term2[0][0]))
        
        dif = X_test[i] - mu[1]
        dif = dif.reshape(-1, 1)
        term2 = np.exp(-0.5 * dif.T * (1/sigma[1]) * dif)
        print('P(C2|x): %f x %f = %f' % (term1[1], term2[0][0], term1[1] * term2[0][0]))
        if i > 5:
            break

    sys.exit(1)

def main():
    # features.shape = (32561, 106)
    features, labels = read_provided_features(sys.argv[3], sys.argv[4])

    # test.shape = (16281, 106)
    tests = read_test_features(sys.argv[5])

    predictions = maximum_likelihood(features, labels, tests)

    with open(sys.argv[6], 'w') as output_file:
        outputs = ['id,label']
        for i, res in enumerate(results):
            outputs.append('%d,%d' % (i+1, res))
        outputs.append('')
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()
