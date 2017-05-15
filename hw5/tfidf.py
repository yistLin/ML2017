#!/usr/local/bin/python3
import nltk
import pickle
import numpy as np
from utils import DataReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def train_tfidf(tokens_list):
    sentences = [' '.join(tokens) for tokens in tokens_list]
    count_vect = CountVectorizer().fit(sentences)
    X_train_counts = count_vect.transform(sentences)
    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    data_vec = tfidf_transformer.transform(X_train_counts)
    print('data_vec.shape =', data_vec.shape)

    with open('counter.pkl', 'wb') as f:
        pickle.dump(count_vect, f)
    with open('transformer.pkl', 'wb') as f:
        pickle.dump(tfidf_transformer, f)

    return data_vec


def train_svm(data_vec, tags, model_path='tfidf-model.pkl'):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC, LinearSVC
    from sklearn.metrics import f1_score

    mlb = MultiLabelBinarizer()
    tag_vec = mlb.fit_transform(tags)
    print('tag_vec.shape =', tag_vec.shape)
    print('mlb.classes_ =', list(mlb.classes_))

    X_train = data_vec
    Y_train = tag_vec
    # X_train, X_valid = data_vec[:-400], data_vec[-400:]
    # Y_train, Y_valid = tag_vec[:-400], tag_vec[-400:]
    # X_train, X_valid = data_vec[400:], data_vec[:400]
    # Y_train, Y_valid = tag_vec[400:], tag_vec[:400]

    clf = OneVsRestClassifier(LinearSVC(C=0.001, class_weight='balanced'), n_jobs=-1)
    # clf = OneVsRestClassifier(SVC(kernel='rbf', C=1, class_weight='balanced'), n_jobs=1)
    clf.fit(X_train, Y_train)
    print('clf.fit() done')

    train_pred = clf.predict(X_train)
    print('[train] f1_score =', f1_score(Y_train, train_pred, average='micro'))
    # valid_pred = clf.predict(X_valid)
    # print('[valid] f1_score =', f1_score(Y_valid, valid_pred, average='micro'))

    with open('mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)


def predict(tokens_list, output_path, model_path='tfidf-model.pkl'):
    with open('counter.pkl', 'rb') as f:
        counter = pickle.load(f)
    with open('transformer.pkl', 'rb') as f:
        transformer = pickle.load(f)
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    with open(model_path, 'rb') as f:
        svm_model = pickle.load(f)

    sentences = [' '.join(tokens) for tokens in tokens_list]

    count_vec = counter.transform(sentences)
    test_vec = transformer.transform(count_vec)
    predictions = svm_model.predict(test_vec)
    results = mlb.inverse_transform(predictions)

    with open(output_path, 'w') as out_f:
        outputs = ['"id","tags"\n']
        for idx, tags in enumerate(results):
            outputs.append('"{}","{}"\n'.format(idx, ' '.join(tags)))
        out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TFIDF')
    parser.add_argument('--train', help='predict')
    parser.add_argument('--predict', help='doc2vec')
    parser.add_argument('--model')
    args = parser.parse_args()

    reader = DataReader()

    if args.train:
        _, tags, texts = reader.read_data(args.train)
        data_vec = train_tfidf(texts)
        model_path = args.model or 'tfidf-model.pkl'
        train_svm(data_vec, tags, model_path=model_path)

    if args.predict:
        _, texts = reader.read_test_data(args.predict)
        model_path = args.model or 'tfidf-model.pkl'
        predict(texts, 'output.csv', model_path=model_path)

