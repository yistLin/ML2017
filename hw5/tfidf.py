#!/usr/local/bin/python3
import pickle
import numpy as np
from utils import DataReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


class Classifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=20000, sublinear_tf=True)
        self.binarizer = MultiLabelBinarizer()

    def train_tfidf(self, sentences):
        self.vectorizer.fit(sentences)

    def train_svm(self, sentences, tags, model_path, nb_cv=0):
        X_train = self.vectorizer.transform(sentences)
        Y_train = self.binarizer.fit_transform(tags)
        print('data_vec.shape =', X_train.shape)
        print('tag_vec.shape =', Y_train.shape)

        est = LinearSVC(C=0.0005, class_weight='balanced')
        self.clf = OneVsRestClassifier(est, n_jobs=1)

        if nb_cv > 0:
            scores = cross_val_score(self.clf, X_train, Y_train, scoring='f1_samples', cv=nb_cv, n_jobs=-1)
            print('[CV] scores =', scores)
            print('[CV]   mean =', scores.mean())
            print('[CV]    std =', scores.std())

        self.clf.fit(X_train, Y_train)

    def predict(self, sentences, output_path):
        test_vec = self.vectorizer.transform(sentences)
        preds = self.clf.predict(test_vec)
        results = self.binarizer.inverse_transform(preds)
    
        with open(output_path, 'w') as out_f:
            outputs = ['"id","tags"\n']
            for idx, tags in enumerate(results):
                alltags = ' '.join(tags)
                outputs.append('"{}","{}"\n'.format(idx, alltags))
            out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('--train', help='predict')
    parser.add_argument('--predict', help='doc2vec')
    parser.add_argument('--model', default='tfidf-model.pkl')
    parser.add_argument('--cv', type=int, default=0)
    parser.add_argument('--output', default='output.csv')
    args = parser.parse_args()

    reader = DataReader()
    model_path = args.model

    if args.train:
        _, tags, texts = reader.read_data(args.train)

        clf = Classifier()
        clf.train_tfidf(texts)
        clf.train_svm(texts, tags, model_path, nb_cv=args.cv)

        with open(model_path, 'wb') as model_f:
            pickle.dump(clf, model_f)

    if args.predict:
        _, texts = reader.read_test_data(args.predict)
        with open(model_path, 'rb') as model_f:
            clf = pickle.load(model_f)
        clf.predict(texts, args.output)

