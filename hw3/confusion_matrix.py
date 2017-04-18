import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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

def predict(model, features):
    X = features / 255
    X.shape = (-1, 48, 48, 1)
    return model.predict_classes(X)

def main():
    labels, features = read_features(sys.argv[1])
    model = load_model(sys.argv[2])
    results = predict(model, features)
    print(','.join(results), file='predict_result.csv')
    class_names = ['0', '1', '2', '3', '4', '5', '6']

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(results, labels)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
    #                       title='Normalized confusion matrix')

    plt.show()

if __name__ == '__main__':
    main()
