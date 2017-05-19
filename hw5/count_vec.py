#!/usr/local/bin/python3
import pickle
import numpy as np
from utils import DataReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score


class Classifier:
    def __init__(self):
        pass

    def train_tfidf(self, sentences):
        self.vectorizer = CountVectorizer(stop_words='english')
        self.vectorizer.fit(sentences)

    def train_svm(self, sentences, tags, model_path, nb_cv=0):
        self.data_vec = self.vectorizer.transform(sentences)
        self.binarizer = MultiLabelBinarizer()
        self.tag_vec = self.binarizer.fit_transform(tags)
        print('data_vec.shape =', self.data_vec.shape)
        print('tag_vec.shape =', self.tag_vec.shape)
        print('mlb.classes_ =', list(self.binarizer.classes_))

        X_train = self.data_vec
        Y_train = self.tag_vec

        est = LinearSVC(C=0.01, class_weight='balanced')
        self.clf = OneVsRestClassifier(est, n_jobs=-1)
        
        if nb_cv > 0:
            scores = cross_val_score(self.clf, X_train, Y_train, scoring='f1_micro', cv=nb_cv, n_jobs=-1)
            print('[CV] scores =', scores)
            print('[CV]   mean =', scores.mean())
            print('[CV]    std =', scores.std())

        self.clf.fit(X_train, Y_train)

    def predict(self, sentences, output_path):
        test_vec = self.vectorizer.transform(sentences)
        predictions = self.clf.predict(test_vec)
        results = self.binarizer.inverse_transform(predictions)
    
        with open(output_path, 'w') as out_f:
            outputs = ['"id","tags"\n']
            for idx, tags in enumerate(results):
                alltags = 'FICTION NOVEL FANTASY' if len(tags) == 0 else ' '.join(tags)
                outputs.append('"{}","{}"\n'.format(idx, alltags))
            out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('--train', help='predict')
    parser.add_argument('--predict', help='doc2vec')
    parser.add_argument('--model', default='tfidf-model.pkl')
    parser.add_argument('--cv', type=int, default=0)
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
        clf.predict(texts, 'output.csv')

