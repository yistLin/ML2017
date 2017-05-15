#!/usr/local/bin/python3
import nltk
import pickle
import numpy as np
import keras.backend as K
from utils import read_data, read_test_data
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from rnn import precision, recall, f1_score


class LabeledLineSentence():
    def __init__(self, texts):
        self.texts = texts

    def __iter__(self):
        for uid, text in enumerate(self.texts):
            yield LabeledSentence(words=text, tags=[str(uid)])


def train_doc2vec(texts, testtexts):
    it = LabeledLineSentence(texts + testtexts)
    
    model = Doc2Vec(size=256, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025, negative=10)
    model.build_vocab(it)

    print('model.corpus_count =', model.corpus_count)
    print('model.iter =', model.iter)
    model.train(it, total_examples=model.corpus_count, epochs=10)

    model.save('doc2vec.model')


def train_clf_doc2vec(tags, texts):
    from sklearn.preprocessing import MultiLabelBinarizer
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import SVC, LinearSVC

    doc2vec = Doc2Vec.load('doc2vec.model')
    text_vec = np.array([doc2vec.docvecs[str(uid)] for uid, _ in enumerate(texts)])
    print('text_vec.shape =', text_vec.shape)

    mlb = MultiLabelBinarizer()
    tag_vec = mlb.fit_transform(tags)
    print('tag_vec.shape =', tag_vec.shape)
    print('mlb.classes_ =', list(mlb.classes_))

    X_train = text_vec
    Y_train = tag_vec
    # X_train, X_valid = text_vec[:-400], text_vec[-400:]
    # Y_train, Y_valid = tag_vec[:-400], tag_vec[-400:]
    # X_train, X_valid = text_vec[400:], text_vec[:400]
    # Y_train, Y_valid = tag_vec[400:], tag_vec[:400]

    # clf = OneVsRestClassifier(LinearSVC(class_weight='balanced'), n_jobs=-1)
    clf = OneVsRestClassifier(SVC(kernel='rbf', C=100, class_weight='balanced'), n_jobs=-1)
    clf.fit(X_train, Y_train)

    train_pred = clf.predict(X_train)
    # valid_pred = clf.predict(X_valid)
    # print('train_pred =', train_pred)
    # print('valid_pred =', valid_pred)

    with open('mlb.pkl', 'wb') as f:
        pickle.dump(mlb, f)
    with open('svm-model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    from sklearn.metrics import f1_score
    print('[train] f1_score =', f1_score(Y_train, train_pred, average='micro'))
    # print('[valid] f1_score =', f1_score(Y_valid, valid_pred, average='micro'))

    # X_train = text_vec
    # Y_train = tag_vec
    # model = Sequential()
    # model.add(Dense(512, input_shape=(300,), activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(38, activation='sigmoid'))
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1_score])

    # model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_split=0.1)


def predict(sentences, output_path):
    doc2vec = Doc2Vec.load('doc2vec.model')
    with open('mlb.pkl', 'rb') as f:
        mlb = pickle.load(f)
    with open('svm-model.pkl', 'rb') as f:
        svm_model = pickle.load(f)

    X = np.array([doc2vec.docvecs[str(4964+uid)] for uid, _ in enumerate(sentences)])
    predictions = svm_model.predict(X)
    results = mlb.inverse_transform(predictions)

    with open(output_path, 'w') as out_f:
        outputs = ['id,tags\n']
        for idx, tags in enumerate(results):
            outputs.append('{},"{}"\n'.format(idx, ','.join(tags)))
        out_f.write(''.join(outputs))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RNN tag predictor.')
    parser.add_argument('data_path', help='images path')
    parser.add_argument('--predict', help='predict', action='store_true')
    parser.add_argument('--doc2vec', help='doc2vec', action='store_true')
    parser.add_argument('--clf_doc2vec', action='store_true')
    args = parser.parse_args()

    # max sequence length is 365
    if args.predict:
        _, texts = read_test_data(args.data_path)
        predict(texts, 'output.csv')
    elif args.doc2vec:
        _, tags, texts = read_data(args.data_path)
        _, testtexts = read_test_data('test_data.csv')
        train_doc2vec(texts, testtexts)
    elif args.clf_doc2vec:
        _, tags, texts = read_data(args.data_path)
        train_clf_doc2vec(tags, texts)

