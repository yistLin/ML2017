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
            labels.append([label])

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
    # return 1.0 / (1.0 + np.clip(np.exp(-z), 1e-3, 1e3))
    return 1.0 / (1.0 + np.exp(-z))

def cross_entropy(y, h):
    loss = -(y * np.log(h + 1e-20) + (1 - y) * np.log(1 - h + 1e-20))
    return np.sum(loss)

def predict(X, theta, X_mean, X_std):
    # X = np.concatenate((X, X**3), axis=1)
    log_age = np.log(X[:,0]).reshape(-1, 1)
    X = np.concatenate((log_age, X[:,1:], X**3, X**5), axis=1)
    X = (X - X_mean) / X_std
    X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)
    h = sigmoid(X.dot(theta))
    return np.where(h > 0.5, 1, 0)

def gradient_descent(X, Y, nb_epoch):
    # settings
    lr = 0.5
    reg_lambda = 0.005

    # add cubic terms
    log_age = np.log(X[:,0]).reshape(-1, 1)
    X = np.concatenate((log_age, X[:,1:], X**3, X**5), axis=1)
    # X = np.concatenate((X, X**3), axis=1)

    # normalization
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X = (X - X_mean) / X_std

    # add bias dimension
    X = np.concatenate((X, np.ones(shape=(len(X), 1))), axis=1)

    # seperate train and valid
    X_valid = X[-6000:]
    Y_valid = Y[-6000:]
    X = X[:-6000]
    Y = Y[:-6000]

    # initial weights and learning rate
    nb_datas, nb_feats = X.shape
    theta = np.zeros(shape=(nb_feats, 1))
    theta_lr = np.zeros(shape=(nb_feats, 1))

    for i in range(nb_epoch):
        h = sigmoid(X.dot(theta))
        delta = h - Y
        grad = ((delta.T).dot(X)).T

        # adaptive gradient descent
        theta_lr = theta_lr + grad ** 2

        # without regularization
        theta = theta - lr / np.sqrt(theta_lr) * grad
        # L1 regularization
        # theta = theta - lr / np.sqrt(theta_lr) * grad - (lr * reg_lambda) * np.where(theta > 0, 1.0, -1.0)
        # L2 regularization
        # theta = (1 - lr * reg_lambda) * theta - lr / np.sqrt(theta_lr) * grad

        if (i+1) % 100 == 0:
            predictions = np.where(h > 0.5, 1, 0)
            accuracy = sum(predictions == Y) / nb_datas

            valid_h = sigmoid(X_valid.dot(theta))
            valid_predictions = np.where(valid_h > 0.5, 1, 0)
            valid_accuracy = sum(valid_predictions == Y_valid) / len(Y_valid)

            print('Epoch [%d]' % (i+1))
            print('  --> train accuracy = %f' % (accuracy))
            print('  --> train loss     = %f' % (cross_entropy(Y, h)/nb_datas))
            print('  --> valid accuracy = %f' % (valid_accuracy))
            print('  --> valid loss     = %f' % (cross_entropy(Y_valid, valid_h)/len(Y_valid)))
            print()

    return theta, X_mean, X_std

def main():
    # features.shape = (32561, 106)
    features, labels = read_provided_features(sys.argv[3], sys.argv[4])

    theta, X_mean, X_std = gradient_descent(features, labels, 2500)

    # test.shape = (16281, 106)
    tests = read_test_features(sys.argv[5])

    # predict test
    results = predict(tests, theta, X_mean, X_std)

    outputs = ['id,label']
    for i, res in enumerate(results):
        outputs.append('%d,%d' % (i+1, res))
    outputs.append('')
    with open(sys.argv[6], 'w') as output_file:
        output_file.write('\n'.join(outputs))

if __name__ == '__main__':
    main()
