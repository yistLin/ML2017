#!/usr/local/bin/python3
import sys
import numpy as np

def read_provided_features(x_filename, y_filename):
    features = []
    with open(x_filename, 'r') as x_file:
        next(x_file) # skip the first row
        for line in x_file:
            fields = line.strip().split(',')
            # features.append(list(map(float, fields)))
            features.append([-1.0 if x == '0' else float(x) for x in fields])

    labels = []
    with open(y_filename, 'r') as y_file:
        for line in y_file:
            labels.append([float(line.strip())])

    return np.array(features), np.array(labels)

def read_test_features(test_filename):
    tests = []
    with open(test_filename, 'r') as test_file:
        next(test_file) # skip the first row
        for line in test_file:
            fields = line.strip().split(',')
            # tests.append(list(map(float, fields)))
            tests.append([-1.0 if x == '0' else float(x) for x in fields])

    return np.array(tests)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cross_entropy(y, h):
    loss = -(y * np.log(h + 1e-20) + (1 - y) * np.log(1 - h + 1e-20))
    return np.sum(loss)

def predict(X, theta, X_max, X_min):
    # X = 2.0 * (X - X_min) / (X_max - X_min + 1e-20) - 1.0
    X = np.concatenate((X, X**3), axis=1)
    X = (X - X_max) / X_min
    X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)
    h = sigmoid(X.dot(theta))
    return np.where(h > 0.5, 1, 0)

def gradient_descent(X, Y, nb_epoch):
    # split valid set
    # perm = np.random.permutation(len(X))
    # X, Y = X[perm], Y[perm]
    # valid_X = X[:6000]
    # X = X[6000:]
    # valid_Y = Y[:6000]
    # Y = Y[6000:]

    # settings
    nb_datas = X.shape[0]
    lr = 0.5
    X = np.concatenate((X, X**3), axis=1)

    # normalization
    # X_max = np.max(X, axis=0)
    # X_min = np.min(X, axis=0)
    # X = (X - X_min) / (X_max - X_min + 1e-20)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # add bias dimension
    X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)
    nb_feats = X.shape[1]

    # initial weights and learning rate
    theta = np.zeros(shape=(nb_feats, 1))
    theta_lr = np.zeros(shape=(nb_feats, 1))

    for i in range(nb_epoch):
        h = sigmoid(X.dot(theta))
        delta = h - Y
        grad = ((delta.T).dot(X)).T
        theta_lr = theta_lr + grad ** 2
        theta = theta - lr / np.sqrt(theta_lr) * grad
        if (i+1) % 100 == 0:
            predictions = np.where(h > 0.5, 1, 0)
            accuracy = sum(predictions == Y) / nb_datas
            print('[%d] train accuracy = %f' % (i+1, accuracy))
            print('[%d] train loss     = %f' % (i+1, cross_entropy(Y, h)/nb_datas))
            # predictions = predict(valid_X, theta, X_mean, X_std)
            # accuracy = sum(predictions == valid_Y) / len(valid_X)
            # print('[%d] valid accuracy = %f' % (i+1, accuracy))
            print()

    return theta, X_mean, X_std

def main():
    # features.shape = (32561, 106)
    features, labels = read_provided_features(sys.argv[3], sys.argv[4])

    theta, X_max, X_min = gradient_descent(features, labels, 600)

    # test.shape = (16281, 106)
    tests = read_test_features(sys.argv[5])

    # predict test
    results = predict(tests, theta, X_max, X_min)

    outputs = ['id,label']
    for i, res in enumerate(results):
        outputs.append('%d,%d' % (i+1, res))
    outputs.append('')
    with open(sys.argv[6], 'w') as output_file:
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()
