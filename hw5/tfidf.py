#!/usr/local/bin/python3
import pickle
import numpy as np
from utils import DataReader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


class Classifier:
    def __init__(self):
        pass

    def train_tfidf(self, sentences):
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 3), max_features=20000, sublinear_tf=True)
        self.vectorizer.fit(sentences)

        # self.counter = CountVectorizer(stop_words='english', ngram_range=(1, 3), max_features=40000)
        # self.transformer = TfidfTransformer(sublinear_tf=True)
        # count_vec = self.counter.fit_transform(sentences)
        # self.transformer.fit(count_vec)

    def train_svm(self, sentences, tags, model_path, nb_cv=0):
        # X_temp = self.counter.transform(sentences)
        # X_train = self.transformer.transform(X_temp)
        X_train = self.vectorizer.transform(sentences)
        self.binarizer = MultiLabelBinarizer()
        Y_train = self.binarizer.fit_transform(tags)
        print('data_vec.shape =', X_train.shape)
        print('tag_vec.shape =', Y_train.shape)

        # X_train, X_valid = X_train[400:], X_train[:400]
        # Y_train, Y_valid = Y_train[400:], Y_train[:400]
        # X_train, X_valid = X_train[:-400], X_train[-400:]
        # Y_train, Y_valid = Y_train[:-400], Y_train[-400:]

        est = LinearSVC(C=0.0005, class_weight='balanced')
        self.clf = OneVsRestClassifier(est, n_jobs=1)

        if nb_cv > 0:
            scores = cross_val_score(self.clf, X_train, Y_train, scoring='f1_samples', cv=nb_cv, n_jobs=-1)
            print('[CV] scores =', scores)
            print('[CV]   mean =', scores.mean())
            print('[CV]    std =', scores.std())

        self.clf.fit(X_train, Y_train)

        # predict and calculate f1_score
        # predictions = self.clf.predict(X_valid)
        # print('[PRED] f1_score (tfidf) =', f1_score(Y_valid, predictions, average='samples'))

    def predict(self, sentences, output_path):
        test_vec = self.vectorizer.transform(sentences)
        # temp_vec = self.counter.transform(sentences)
        # test_vec = self.transformer.transform(temp_vec)
        preds = self.clf.predict(test_vec)
        results = self.binarizer.inverse_transform(preds)
    
        with open(output_path, 'w') as out_f:
            outputs = ['"id","tags"\n']
            for idx, tags in enumerate(results):
                alltags = 'FICTION NOVEL' if len(tags) == 0 else ' '.join(tags)
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

